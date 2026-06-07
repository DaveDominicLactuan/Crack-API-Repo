import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image  # 1. Import Pillow
import sys
import subprocess
import cv2
import numpy as np
from scipy import stats
import uuid
import base64

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
# Setup Upload Directory
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Mount Static Files


# Mount the uploads folder so files can be accessed via URL (e.g., http://localhost:8000/uploads/filename.jpg)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/assessed", StaticFiles(directory=ASSESSED_DIR), name="assessed")


PIXEL_TO_MM = 0.1

# Function 1: Simple Text Endpoint
@app.get("/api/hello")
async def get_message():
    return {"message": "Hello from your FastAPI backend! 🚀"}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1. Save the raw image from the Ionic app
    raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(raw_file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 2. Process the image
    # We receive 3 values, 2 images the cleaned_mask, visual_img and bounding_box data
    # We must unpack the TWO images returned by our OpenCV function
    cleaned_mask, visual_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64 = preprocess_and_save_format(raw_file_path)
    

    # Print to the console to check the received bounding box
    print(f"--- API ROUTE CHECK --- Bounding Boxes received: {bounding_boxes}")

    # cv2.imshow("Cleaned Mask", cleaned_mask)  # Debug: Show the cleaned mask to verify it's correct
    
    # 3. Save the newly processed image to the processed directory
    processed_filename = f"processed_{file.filename}"
    processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
    
    # Save the 'visual_img' with the green boxes and red dots
    # so the user can see the results in the folders
    cv2.imwrite(processed_file_path, visual_img)
    
    # (Optional) save the pure black and white mask:
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
        "bounding_boxes": bounding_boxes, # This is the new structured data about the boxes for Ionic to use
        "cropped_roi_objectStore": cropped_roi_objectStore, # This is the new structured data about the cropped ROIs for Ionic to use
        "resizedImagePath": resized_img_base64 # This is the Base64 string for the resized image for Ionic to use
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



def preprocess_and_save_format(image_path):
    # generate the txt path matching the image name for bug testing
    # example "uploads/crack_001.jpg" becomes "uploads/crack_001.txt"
    base_path, _ = os.path.splitext(image_path)
    txt_path = f"{base_path}.txt"
    
    # Read the raw image saved by FastAPI
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to a standard size (e.g., 416x416) for consistent processing
    resized_img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    # Blurr the image with gaussian Blurr to reduce noise before thresholding
    blurred = cv2.GaussianBlur(resized_img, (3, 3), 0)
    # cv2.imshow("Resized", resized_img)  # Debug: Show the blurred image to verify it's correct
    resized_img_base64 = convert_to_base64(resized_img)

    # Converts the grayscale image to a binary bitmap using adaptive thresholding
    whitehot_crack = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=5
    )
    
    # this function connects the dots/pixels and groups touching white pixels together
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        whitehot_crack, connectivity=8, ltype=cv2.CV_32S
    )
    
    # cleaned image will hold only the valid crack components after filtering out small noise
    cleaned_image = np.zeros_like(whitehot_crack)
    # We create a BGR version of the resized grayscale image to draw colored boxes and circles on it and store
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
                txt_file.write(f"{w} {h} {x} {y}\n")
                
                # 2. Append the dictionary to our bounding_boxes list
                bounding_boxes.append({
                    "w": int(w), 
                    "h": int(h), 
                    "x": int(x), 
                    "y": int(y)
                })

    print(f"--- OPENCV CHECK --- Initial bounding_boxes array: {bounding_boxes}")

    # --- ADDED SORTING LOGIC ---
    # Sort the bounding boxes from largest to smallest based on area (width * height)
    bounding_boxes.sort(key=lambda box: box["w"] * box["h"], reverse=True)
    
    print(f"--- OPENCV CHECK --- Sorted bounding_boxes array: {bounding_boxes}")

    # =========================================================================
    # --- CROPPING, BASE64 ENCODING & PIXEL ANALYSIS ---
    # =========================================================================
    
    # Initialize the new object store for the cropped images
    cropped_roi_objectStore = []

    # Create a fresh BGR representation of the resized image
    image = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    image_with_box = image.copy()

    # --- LOOP: Process every box in the array ---
    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        
        # Draw the rectangle on the main image
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 1. Crop the clean region of interest
        cropped_roi = image[y:y+h, x:x+w]

        # 2. Encode to Base64 and save to cropped_roi_objectStore
        success, buffer = cv2.imencode('.jpg', cropped_roi)
        
        if success:
            img_base64_str = base64.b64encode(buffer).decode('utf-8')
            image_data_uri = f"data:image/jpeg;base64,{img_base64_str}"
        else:
            image_data_uri = None

        # Append to the separate store, using 'box_id' to link it back to the box
        cropped_roi_objectStore.append({
            "box_id": i,
            "image_data": image_data_uri
        })

        # 3. Convert to Grayscale for pixel analysis
        gray_cropped = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
        
        # 4. Convert to Binary Bitmap
        _, bitmap_cropped = cv2.threshold(gray_cropped, 127, 255, cv2.THRESH_BINARY)

        # --- PIXEL ANALYSIS ---
        total_pixels = bitmap_cropped.size
        white_pixels = cv2.countNonZero(bitmap_cropped)
        black_pixels = total_pixels - white_pixels
        
        if total_pixels > 0:
            white_percentage = (white_pixels / total_pixels) * 100
            black_percentage = (black_pixels / total_pixels) * 100

            # Print the results to the console
            print(f"\n--- Image Pixel Analysis (Box {i}) ---")
            print(f"Total Pixels: {total_pixels:,}")
            print(f"White Pixels (Background): {white_pixels:,} ({white_percentage:.2f}%)")
            print(f"Black Pixels (Crack):     {black_pixels:,} ({black_percentage:.2f}%)")

            # Update the bounding box dictionary with pixel analysis data
            box["total_pixels"] = int(total_pixels)
            box["white_pixels"] = int(white_pixels)
            box["white_percentage"] = round(white_percentage, 2)
            box["black_pixels"] = int(black_pixels)
            box["black_percentage"] = round(black_percentage, 2)

    # 5. Return the FOUR required items
    return cleaned_image, output_img, bounding_boxes, cropped_roi_objectStore, resized_img_base64

@app.post("/api/upload2")
async def upload_image2(file: UploadFile = File(...)):
    # FORCE a clean filename on the server to ignore whatever the client sends
    # This bypasses the 'filename' property corruption entirely
    safe_filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Use shutil to stream directly to disk
        import shutil
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success",
            "filename": safe_filename,
            "url": f"/uploads/{safe_filename}"
        }
    except Exception as e:
        # Log the actual incoming filename to your server console for debugging
        print(f"DEBUG: Incoming filename was: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_to_base64(image):
    """Converts an OpenCV image (numpy array) to a Base64 string."""
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        img_base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64_str}"
    return None