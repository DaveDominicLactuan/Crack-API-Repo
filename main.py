import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from scipy import stats
import uuid
import base64
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
ASSESSED_DIR = "assessed_cracks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ASSESSED_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/assessed", StaticFiles(directory=ASSESSED_DIR), name="assessed")

# --- REAL WORLD MATH CONSTANTS ---
PIXEL_TO_MM = 0.1
GAP_THRESHOLD_PIXELS = 5

# --- UNION FIND CLASS FOR GROUPING CLOSE CRACKS ---
class CrackUnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def calculate_min_edge_distance(contour_a, contour_b):
    pts_a = contour_a.reshape(-1, 2)
    pts_b = contour_b.reshape(-1, 2)
    dist_matrix = np.linalg.norm(pts_a[:, np.newaxis] - pts_b, axis=2)
    return np.min(dist_matrix)

def convert_to_base64(image):
    """Converts an OpenCV image (numpy array) to a Base64 string."""
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        img_base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64_str}"
    return None

@app.get("/api/hello")
async def get_message():
    return {"message": "Hello from your FastAPI backend 🚀"}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(raw_file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    cleaned_mask, visual_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64 = preprocess_and_save_format(raw_file_path)
    
    print(f"--- API ROUTE CHECK --- Bounding Boxes received: {bounding_boxes}")
    
    processed_filename = f"processed_{file.filename}"
    processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
    cv2.imwrite(processed_file_path, visual_img)
    
    return {
        "message": "Image analyzed successfully!",
        "rawImagePath": f"/uploads/{file.filename}",
        "processedImagePath": f"/processed/{processed_filename}",
        "bounding_boxes": bounding_boxes,
        "cropped_roi_objectStore": cropped_roi_objectStore,
        "resizedImagePath": resized_img_base64 
    }

def preprocess_and_save_format(image_path):
    target_w, target_h = 416, 416
    
    # 1. Read and Resize (Keep original clean)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Export unmodified base64 image immediately
    resized_img_base64 = convert_to_base64(resized_img)
    
    # Create copies for processing and drawing
    output_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    clean_roi_source = output_img.copy() # Pristine color copy specifically for cropping

    # 2. Advanced Filtering and Thresholding
    smoothed_small = cv2.bilateralFilter(resized_img, d=7, sigmaColor=50, sigmaSpace=50)
    whitehot_crack = cv2.adaptiveThreshold(
        smoothed_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 14
    )

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    cleaned_mask = cv2.morphologyEx(whitehot_crack, cv2.MORPH_OPEN, kernel_small)

    num_labels, labels, stats_map, centroids = cv2.connectedComponentsWithStats(
        cleaned_mask, connectivity=8, ltype=cv2.CV_32S
    )
    
    final_cleaned_mask = np.zeros_like(cleaned_mask)
    for i in range(1, num_labels):
        if stats_map[i, cv2.CC_STAT_AREA] >= 75:
            final_cleaned_mask[labels == i] = 255

    # 3. Contour Extraction and Math Analysis
    contours, _ = cv2.findContours(final_cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crack_records = []

    for count in contours:
        if cv2.contourArea(count) < 5:
            continue
    
        x, y, w, h = cv2.boundingRect(count)
        region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8)
        shifted_count = count - [x - 5, y - 5]
        cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1)
        
        # Skeletonization for real-world measurements
        dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5)
        skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        
        pixel_length = np.sum(skeleton == 255)
        length_mm = pixel_length * PIXEL_TO_MM

        if pixel_length == 0:
            continue

        widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
        max_w = np.max(widths_mm)
        mean_w = np.mean(widths_mm)
        rotated_box = cv2.minAreaRect(count)
        orientation_angle = float(rotated_box[2])

        crack_records.append({
            "contour": count,
            "length_mm": length_mm,
            "max_width_mm": max_w,
            "mean_width_mm": mean_w,
            "orientation_deg": orientation_angle,
            "widths_raw": widths_mm
        })

    num_fragments = len(crack_records)
    
    bounding_boxes = []
    cropped_roi_objectStore = []

    if num_fragments > 0:
        # 4. Group Nearby Cracks
        uf = CrackUnionFind(num_fragments)
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                gap = calculate_min_edge_distance(crack_records[i]["contour"], crack_records[j]["contour"])
                if gap <= GAP_THRESHOLD_PIXELS:
                    uf.union(i, j)

        clusters = {}
        for i in range(num_fragments):
            root = uf.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        # 5. Process Grouped Clusters
        for chain_id, indices in enumerate(clusters.values()):
            sub_cracks = [crack_records[idx] for idx in indices]
            total_length = sum(c["length_mm"] for c in sub_cracks)
            max_width = max(c["max_width_mm"] for c in sub_cracks)
            mean_width = np.mean(np.concatenate([c["widths_raw"] for c in sub_cracks]))
            angles = np.array([c["orientation_deg"] for c in sub_cracks])
            lengths = np.array([c["length_mm"] for c in sub_cracks])

            if total_length > 0:
                percentage_weights = lengths / total_length
                weighted_average_angle = np.sum(angles * percentage_weights)
                within_tolerance = np.abs(angles - weighted_average_angle) <= 10.0

                if np.sum(within_tolerance) > (len(sub_cracks) / 2.0):
                    final_orientation = f"{weighted_average_angle:.1f} degrees"
                else:
                    final_orientation = "Curve"
            else:
                final_orientation = "Unknown"

            all_contour_points = np.concatenate([c["contour"] for c in sub_cracks], axis=0)

            # Get Master Bounding Box for the grouped crack
            bx, by, bw, bh = cv2.boundingRect(all_contour_points)
            
            # Ensure crop boundaries don't exceed image dimensions
            crop_y1, crop_y2 = max(0, by), min(target_h, by + bh)
            crop_x1, crop_x2 = max(0, bx), min(target_w, bx + bw)
            
            # Crop ROI from the pristine copy
            cropped_roi = clean_roi_source[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Encode cropped region to Base64
            image_data_uri = convert_to_base64(cropped_roi)
            
            cropped_roi_objectStore.append({
                "box_id": chain_id,
                "image_data": image_data_uri
            })

            # Draw visual indicators on output_img
            cv2.rectangle(output_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            # Find center of bounding box to draw red dot
            cx = bx + (bw // 2)
            cy = by + (bh // 2)
            cv2.circle(output_img, (cx, cy), 4, (0, 0, 255), -1)

            # Append the data package formatted for the frontend
            bounding_boxes.append({
                "id": chain_id,
                "x": int(bx),
                "y": int(by),
                "w": int(bw),
                "h": int(bh),
                "crackLength_mm": round(total_length, 2),
                "avgWidth_mm": round(mean_width, 2),
                "maxWidth_mm": round(max_width, 2),
                "orientation": final_orientation
            })

    # Sort bounding boxes by length (largest to smallest)
    bounding_boxes.sort(key=lambda box: box["crackLength_mm"], reverse=True)

    return final_cleaned_mask, output_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)