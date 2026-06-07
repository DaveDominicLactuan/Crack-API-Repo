import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image  # 1. Import Pillow
import sys
import subprocess
import cv2
import numpy as np
import os
from scipy import stats


app = FastAPI()

# CRITICAL: Enable CORS so your Ionic app can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed" # New folder for the OpenCV outputs
ASSESSED_DIR = "assessed_cracks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ASSESSED_DIR, exist_ok=True)

# Mount the uploads folder so files can be accessed via URL (e.g., http://localhost:8000/uploads/filename.jpg)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/assessed", StaticFiles(directory=ASSESSED_DIR), name="assessed")


PIXEL_TO_MM = 0.1

# Function 1: Simple Text Endpoint
@app.get("/api/hello")
async def get_message():
    return {"message": "Hello from your FastAPI backend! 🚀"}

# --- OLD: Full Pipeline with Analysis and JSON Metrics --- 7:21
# Function 2: Image Upload Endpoint
# @app.post("/api/upload")
# async def upload_image(file: UploadFile = File(...)):
#     # Create a safe path to save the file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     # Save the file locally
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())
        
#     # Return the relative path of the uploaded image back to the app
#     return {
#         "message": "Image uploaded successfully!",
#         "imagePath": f"/uploads/{file.filename}"
#     }

# --- OLD: Full Pipeline with Analysis and JSON Metrics --- 8:30
# @app.post("/api/upload")
# async def upload_image(file: UploadFile = File(...)):
#     # Create a safe path to save the file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     # Save the file locally
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())
        
#     # --- NEW: Open the image in a local GUI window ---
#     try:
#         img = Image.open(file_path)
#         img.show()  # This opens your OS default photo viewer
#     except Exception as e:
#         # Wrapped in a try/except so your API doesn't crash if the GUI fails
#         print(f"Log: Desktop GUI not available to display image. Error: {e}")
#     # -------------------------------------------------
        
#     # Return the relative path of the uploaded image back to the app
#     return {
#         "message": "Image uploaded successfully!",
#         "imagePath": f"/uploads/{file.filename}"
#     }

# --- OLD: Full Pipeline with Analysis and JSON Metrics --- 9:00
# @app.post("/api/upload")
# async def upload_image(file: UploadFile = File(...)):
#     # 1. Save the raw image from the Ionic app
#     raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(raw_file_path, "wb") as buffer:
#         buffer.write(await file.read())
        
#     # 2. Process the image using your OpenCV function
#     processed_img_array = preprocess_and_save_format(raw_file_path)
    
#     # 3. Save the newly processed image to the processed directory
#     processed_filename = f"processed_{file.filename}"
#     processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
#     cv2.imwrite(processed_file_path, processed_img_array)
    
#     # 4. Open the image in a window (Using Native OS viewer so it doesn't crash the server thread)
#     try:
#         if sys.platform == "win32":
#             os.startfile(processed_file_path)  # Windows
#         elif sys.platform == "darwin":
#             subprocess.Popen(["open", processed_file_path])  # macOS
#         else:
#             subprocess.Popen(["xdg-open", processed_file_path])  # Linux
#     except Exception as e:
#         print(f"Log: Could not open system window. Error: {e}")


#         # --- NEW: Open the image in a local GUI window ---
#     try:
#         img = Image.open(raw_file_path)
#         img.show()  # This opens your OS default photo viewer
#     except Exception as e:
#         # Wrapped in a try/except so your API doesn't crash if the GUI fails
#         print(f"Log: Desktop GUI not available to display image. Error: {e}")
#     # -------------------------------------------------
        
#     # Return the relative path of the uploaded image back to the app
#     return {
#         "message": "Image uploaded successfully!",
#         "imagePath": f"/uploads/{file.filename}"
#     }
        
#     # 5. Return BOTH paths back to the Ionic app
#     return {
#         "message": "Image analyzed successfully!",
#         "rawImagePath": f"/uploads/{file.filename}",
#         "processedImagePath": f"/processed/{processed_filename}"
#     }

# --- NEW: Full Pipeline with Analysis and JSON Metrics --- 9:21
# @app.post("/api/upload")
# async def upload_image(file: UploadFile = File(...)):
#     # 1. Save uploaded file
#     raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(raw_file_path, "wb") as buffer:
#         buffer.write(await file.read())
        
#     # 2. Run the analysis pipeline
#     base_name = os.path.splitext(file.filename)
#     mask = load_binary_mask(raw_file_path)
#     records = assess_crack(mask)
    
#     # 3. Generate images and extract clean JSON data
#     bbox_img_path, json_metrics = generate_and_save_outputs(base_name, mask, records, ASSESSED_DIR)
    
#     # 4. Pop open the bounding box image locally
#     try:
#         if sys.platform == "win32":
#             os.startfile(bbox_img_path)
#         elif sys.platform == "darwin":
#             subprocess.Popen(["open", bbox_img_path])
#         else:
#             subprocess.Popen(["xdg-open", bbox_img_path])
#     except Exception as e:
#         print(f"Log: Could not open system window. Error: {e}")

#     # 5. Return everything back to Ionic!
#     return {
#         "message": "Crack analyzed successfully!",
#         "rawImagePath": f"/uploads/{file.filename}",
#         "assessedImagePath": f"/assessed/{base_name}_bbox.jpg",
#         "measurements": json_metrics # Now Ionic can display the length/width on screen
#     }

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1. Save the raw image from the Ionic app
    raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(raw_file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 2. Process the image
    # We must unpack the TWO images returned by our OpenCV function
    cleaned_mask, visual_img, bounding_boxes = preprocess_and_save_format(raw_file_path)
    
    # 3. Save the newly processed image to the processed directory
    processed_filename = f"processed_{file.filename}"
    processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
    
    # Let's save the 'visual_img' (with the green boxes and red dots) 
    # so the user can see the results in the app.
    cv2.imwrite(processed_file_path, visual_img)
    
    # (Optional) If you also want to save the pure black and white mask:
    # mask_filename = f"mask_{file.filename}"
    # cv2.imwrite(os.path.join(PROCESSED_DIR, mask_filename), cleaned_mask)
    
    # 4. Open the image in a window (Using Native OS viewer)
    try:
        if sys.platform == "win32":
            os.startfile(processed_file_path)  # Windows
        elif sys.platform == "darwin":
            subprocess.Popen(["open", processed_file_path])  # macOS
        else:
            subprocess.Popen(["xdg-open", processed_file_path])  # Linux
    except Exception as e:
        print(f"Log: Could not open system window. Error: {e}")
        
    # 5. Return BOTH paths back to the Ionic app
    # (Note: I removed the duplicate early return from your original code)
    return {
        "message": "Image analyzed successfully!",
        "rawImagePath": f"/uploads/{file.filename}",
        "processedImagePath": f"/processed/{processed_filename}",
        "bounding_boxes": bounding_boxes # This is the new structured data about the boxes for Ionic to use
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

#Old 
#     # --- YOUR OPENCV FUNCTION ---
# def preprocess_and_save_format(image_path):
#     # Read the raw image saved by FastAPI
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
#     blurred = cv2.GaussianBlur(resized_img, (3, 3), 0)
    
#     whitehot_crack = cv2.adaptiveThreshold(
#         blurred,
#         maxValue=255,
#         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv2.THRESH_BINARY_INV,
#         blockSize=11,
#         C=5
#     )

#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         whitehot_crack, connectivity=8, ltype=cv2.CV_32S
#     )
    
#     cleaned_image = np.zeros_like(whitehot_crack)

#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] >= 16:
#             cleaned_image[labels == i] = 255

#     return cleaned_image



# # --- YOUR OPENCV FUNCTION ---
# def preprocess_and_save_format(image_path):
#     # Automatically generate the txt path matching the image name
#     # e.g., "uploads/crack_001.jpg" becomes "uploads/crack_001.txt"
#     base_path, _ = os.path.splitext(image_path)
#     txt_path = f"{base_path}.txt"
    
#     # Read the raw image saved by FastAPI
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
#     blurred = cv2.GaussianBlur(resized_img, (3, 3), 0)
    
#     whitehot_crack = cv2.adaptiveThreshold(
#         blurred,
#         maxValue=255,
#         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv2.THRESH_BINARY_INV,
#         blockSize=11,
#         C=5
#     )

#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         whitehot_crack, connectivity=8, ltype=cv2.CV_32S
#     )
    
#     cleaned_image = np.zeros_like(whitehot_crack)
#     output_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

#     # Open the text file for writing box data
#     with open(txt_path, "w") as txt_file:
#         for i in range(1, num_labels):
#             area = stats[i, cv2.CC_STAT_AREA]
            
#             # Filter out small noise
#             if area >= 16:
#                 cleaned_image[labels == i] = 255
                
#                 # Extract structural properties
#                 x = stats[i, cv2.CC_STAT_LEFT]
#                 y = stats[i, cv2.CC_STAT_TOP]
#                 w = stats[i, cv2.CC_STAT_WIDTH]
#                 h = stats[i, cv2.CC_STAT_HEIGHT]
                
#                 # Get the center point coordinates
#                 cx, cy = centroids[i]
                
#                 # Draw green bounding box around foreground components
#                 cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
#                 # Mark the center with a small red circle
#                 cv2.circle(output_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                
#                 # Output metrics to console
#                 print(f"Component {i}: Area = {area}px, Center = ({cx:.1f}, {cy:.1f})")
                
#                 # --- WRITE BOUNDING BOX DATA TO TXT FILE ---
#                 # Option A: Standard space-separated values (Most ML loaders prefer this)
#                 txt_file.write(f"{w} {h} {x} {y}\n")
                
#                 # Option B: If you literally want commas and parentheses, uncomment the line below:
#                 # txt_file.write(f"({w}, {h}, {x}, {y})\n")

#     return cleaned_image, output_img

def preprocess_and_save_format(image_path):
    # Automatically generate the txt path matching the image name
    # e.g., "uploads/crack_001.jpg" becomes "uploads/crack_001.txt"
    base_path, _ = os.path.splitext(image_path)
    txt_path = f"{base_path}.txt"
    
    # Read the raw image saved by FastAPI
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized_img, (3, 3), 0)
    
    whitehot_crack = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=5
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        whitehot_crack, connectivity=8, ltype=cv2.CV_32S
    )
    
    cleaned_image = np.zeros_like(whitehot_crack)
    output_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    # 1. Initialize a list to hold the bounding box objects
    bounding_boxes = []

    # Open the text file for writing box data
    with open(txt_path, "w") as txt_file:
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out small noise
            if area >= 16:
                cleaned_image[labels == i] = 255
                
                # Extract structural properties
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Get the center point coordinates
                cx, cy = centroids[i]
                
                # Draw green bounding box around foreground components
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Mark the center with a small red circle
                cv2.circle(output_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                
                # Output metrics to console
                print(f"Component {i}: Area = {area}px, Center = ({cx:.1f}, {cy:.1f})")
                
                # --- WRITE BOUNDING BOX DATA TO TXT FILE ---
                # Standard space-separated values
                txt_file.write(f"{w} {h} {x} {y}\n")
                
                # 2. Append the dictionary to our bounding_boxes list
                # Casting to int() prevents FastAPI JSON serialization errors
                bounding_boxes.append({
                    "w": int(w), 
                    "h": int(h), 
                    "x": int(x), 
                    "y": int(y)
                })

    # 3. Return the new object alongside the images
    return cleaned_image, output_img, bounding_boxes


PIXEL_TO_MM = 0.1

# --- YOUR ANALYSIS FUNCTIONS ---

def load_binary_mask(file_path, threshold=0.5):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

# def assess_crack(binary_mask):
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     crack_records = []

#     for count in contours:
#         if cv2.contourArea(count) < 5:
#             continue

#         x, y, w, h = cv2.boundingRect(count)
#         region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8)
#         shifted_count = count - [x - 5, y - 5]
#         cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1)
        
#         dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5)
#         skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        
#         pixel_length = np.sum(skeleton == 255)
#         length_mm = pixel_length * PIXEL_TO_MM

#         if pixel_length == 0: continue

#         widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
#         max_w = np.max(widths_mm)
#         mean_w = np.mean(widths_mm)
        
#         # Handle SciPy mode safely across all NumPy versions
#         mode_result = stats.mode(np.round(widths_mm, 2), keepdims=True)
        
#         # Flatten the result
#         mode_array = np.ravel(mode_result.mode)
        
#         # FIX 1: Extract the first item using before calling float()
#        # Handle SciPy mode safely across all NumPy versions
#         mode_result = stats.mode(np.round(widths_mm, 2), keepdims=True)
        
#         # BULLETPROOF FIX: Use .size to check if it's empty, and .item() to safely extract the Python float
#         if mode_result.mode.size > 0:
#             mode_w = mode_result.mode.item() 
#         else:
#             mode_w = 0.0
        
#         # rotated_box = cv2.minAreaRect(count)
#         # # FIX 2: Extract just the angle from the tuple (index 2)
#         # orientation_angle = rotated_box

#         # crack_records.append({
#         #     "length_mm": round(length_mm, 2),
#         #     "max_width_mm": round(max_w, 2),
#         #     "mean_width_mm": round(mean_w, 2),
#         #     "mode_width_mm": round(mode_w, 2),
#         #     "orientation_deg": round(orientation_angle, 1),
#         #     "contour": count,         # Kept for drawing
#         #     "rotated_box": rotated_box # Kept for drawing
#         # })

#         rotated_box = cv2.minAreaRect(count)
        
#         # FIX: Add to extract just the angle from the tuple
#         orientation_angle = rotated_box 

#         crack_records.append({
#             "length_mm": round(length_mm, 2),
#             "max_width_mm": round(max_w, 2),
#             "mean_width_mm": round(mean_w, 2),
#             "mode_width_mm": round(mode_w, 2),
#             "orientation_deg": round(orientation_angle, 1), # Now this is a standard float!
#             "contour": count,         # Kept for drawing
#             "rotated_box": rotated_box # Kept for drawing
#         })

#     return crack_records

def assess_crack(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crack_records = []

    for count in contours:
        if cv2.contourArea(count) < 5:
            continue

        x, y, w, h = cv2.boundingRect(count)
        region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8)
        shifted_count = count - [x - 5, y - 5]
        cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1)
        
        dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5)
        skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        
        pixel_length = np.sum(skeleton == 255)
        length_mm = pixel_length * PIXEL_TO_MM

        if pixel_length == 0: continue

        widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
        max_w = np.max(widths_mm)
        mean_w = np.mean(widths_mm)
        
        # --- FIX 1: Bulletproof SciPy mode extraction ---
        mode_result = stats.mode(np.round(widths_mm, 2), keepdims=True)
        if mode_result.mode.size > 0:
            mode_w = mode_result.mode.item() 
        else:
            mode_w = 0.0
        
        # --- FIX 2: Extract just the angle from the RotatedRect tuple ---
        rotated_box = cv2.minAreaRect(count)
        orientation_angle = float(rotated_box) # Notice the added here!

        # --- Cast all outputs to standard floats to ensure clean JSON serialization ---
        crack_records.append({
            "length_mm": round(float(length_mm), 2),
            "max_width_mm": round(float(max_w), 2),
            "mean_width_mm": round(float(mean_w), 2),
            "mode_width_mm": round(float(mode_w), 2),
            "orientation_deg": round(orientation_angle, 1),
            "contour": count,         # Kept for drawing
            "rotated_box": rotated_box # Kept for drawing
        })
    return crack_records

def generate_and_save_outputs(base_name, binary_mask, records, output_dir):
    img_bbox = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # We will strip out the raw contour/box objects so we can return clean JSON to Ionic
    clean_records_for_json = []

    for idx, crack in enumerate(records):
        box_points = cv2.boxPoints(crack["rotated_box"])
        box_points = np.int64(box_points)
        cv2.drawContours(img_bbox, [box_points], 0, (0, 255, 0), 2)
        
        # Create a dictionary safe for JSON serialization
        clean_records_for_json.append({
            "crack_id": idx,
            "length_mm": crack["length_mm"],
            "max_width_mm": crack["max_width_mm"],
            "mean_width_mm": crack["mean_width_mm"],
            "mode_width_mm": crack["mode_width_mm"],
            "orientation_deg": crack["orientation_deg"]
        })

    bbox_path = os.path.join(output_dir, f"{base_name}_bbox.jpg")
    cv2.imwrite(bbox_path, img_bbox)
    
    return bbox_path, clean_records_for_json

    
PIXEL_TO_MM = 0.1

# --- YOUR ANALYSIS FUNCTIONS ---

def load_binary_mask(file_path, threshold=0.5):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def assess_crack(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crack_records = []

    for count in contours:
        if cv2.contourArea(count) < 5:
            continue

        x, y, w, h = cv2.boundingRect(count)
        region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8)
        shifted_count = count - [x - 5, y - 5]
        cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1)
        
        dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5)
        skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        
        pixel_length = np.sum(skeleton == 255)
        length_mm = pixel_length * PIXEL_TO_MM

        if pixel_length == 0: continue

        widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
        max_w = np.max(widths_mm)
        mean_w = np.mean(widths_mm)
        
        # --- FIX 1: Bulletproof SciPy mode extraction ---
        mode_result = stats.mode(np.round(widths_mm, 2), keepdims=True)
        if mode_result.mode.size > 0:
            mode_w = mode_result.mode.item() 
        else:
            mode_w = 0.0
        
        # --- FIX 2: Extract just the angle from the RotatedRect tuple ---
        rotated_box = cv2.minAreaRect(count)
        orientation_angle = float(rotated_box) # Notice the added here!

        # --- Cast all outputs to standard floats to ensure clean JSON serialization ---
        crack_records.append({
            "length_mm": round(float(length_mm), 2),
            "max_width_mm": round(float(max_w), 2),
            "mean_width_mm": round(float(mean_w), 2),
            "mode_width_mm": round(float(mode_w), 2),
            "orientation_deg": round(orientation_angle, 1),
            "contour": count,         # Kept for drawing
            "rotated_box": rotated_box # Kept for drawing
        })
    return crack_records

def generate_and_save_outputs(base_name, binary_mask, records, output_dir):
    img_bbox = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # We will strip out the raw contour/box objects so we can return clean JSON to Ionic
    clean_records_for_json = []

    for idx, crack in enumerate(records):
        box_points = cv2.boxPoints(crack["rotated_box"])
        box_points = np.int64(box_points)
        cv2.drawContours(img_bbox, [box_points], 0, (0, 255, 0), 2)
        
        # Create a dictionary safe for JSON serialization
        clean_records_for_json.append({
            "crack_id": idx,
            "length_mm": crack["length_mm"],
            "max_width_mm": crack["max_width_mm"],
            "mean_width_mm": crack["mean_width_mm"],
            "mode_width_mm": crack["mode_width_mm"],
            "orientation_deg": crack["orientation_deg"]
        })

    bbox_path = os.path.join(output_dir, f"{base_name}_bbox.jpg")
    cv2.imwrite(bbox_path, img_bbox)
    
    return bbox_path, clean_records_for_json