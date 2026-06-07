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

# @app.post("/api/upload")
# async def upload_image(file: UploadFile = File(...)):
#     # 1. Save the raw image from the Ionic app
#     raw_file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(raw_file_path, "wb") as buffer:
#         buffer.write(await file.read())
        
#     # 2. Process the image
#     # We must unpack the TWO images returned by our OpenCV function
#     cleaned_mask, visual_img, bounding_boxes = preprocess_and_save_format(raw_file_path)
    
#     # 3. Save the newly processed image to the processed directory
#     processed_filename = f"processed_{file.filename}"
#     processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
    
#     # Let's save the 'visual_img' (with the green boxes and red dots) 
#     # so the user can see the results in the app.
#     cv2.imwrite(processed_file_path, visual_img)
    
#     # (Optional) If you also want to save the pure black and white mask:
#     # mask_filename = f"mask_{file.filename}"
#     # cv2.imwrite(os.path.join(PROCESSED_DIR, mask_filename), cleaned_mask)
    
#     # 4. Open the image in a window (Using Native OS viewer)
#     try:
#         if sys.platform == "win32":
#             os.startfile(processed_file_path)  # Windows
#         elif sys.platform == "darwin":
#             subprocess.Popen(["open", processed_file_path])  # macOS
#         else:
#             subprocess.Popen(["xdg-open", processed_file_path])  # Linux
#     except Exception as e:
#         print(f"Log: Could not open system window. Error: {e}")
        
#     # 5. Return BOTH paths back to the Ionic app
#     # (Note: I removed the duplicate early return from your original code)
#     return {
#         "message": "Image analyzed successfully!",
#         "rawImagePath": f"/uploads/{file.filename}",
#         "processedImagePath": f"/processed/{processed_filename}",
#         "bounding_boxes": bounding_boxes # This is the new structured data about the boxes for Ionic to use
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
    
    # --- NEW PRINT STATEMENT HERE ---
    # Print to the console to check the content received by the endpoint
    print(f"--- API ROUTE CHECK --- Bounding Boxes received: {bounding_boxes}")
    
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

#     # 1. Initialize a list to hold the bounding box objects
#     bounding_boxes = []

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
#                 # Standard space-separated values
#                 txt_file.write(f"{w} {h} {x} {y}\n")
                
#                 # 2. Append the dictionary to our bounding_boxes list
#                 # Casting to int() prevents FastAPI JSON serialization errors
#                 bounding_boxes.append({
#                     "w": int(w), 
#                     "h": int(h), 
#                     "x": int(x), 
#                     "y": int(y)
#                 })

#     # 3. Return the new object alongside the images
#     return cleaned_image, output_img, bounding_boxes

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
#     cv2.imshow("Whitehot Crack Mask", whitehot_crack)  # Debug: Show the binary mask to verify it's correct
#     cv2.imshow("Resized Grayscale Image", resized_img)  # Debug: Show the resized grayscale image before processing
#     cv2.imshow("Cleaned Image", cleaned_image)  # Debug: Show the cleaned image to verify it's being updated correctly
#     cv2.imshow("Output Image with Boxes", output_img)  # Debug: Show the output image to verify boxes are drawn correctly

#     # 1. Initialize a list to hold the bounding box objects
#     bounding_boxes = []

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
#                 # Standard space-separated values
#                 txt_file.write(f"{w} {h} {x} {y}\n")
                
#                 # 2. Append the dictionary to our bounding_boxes list
#                 # Casting to int() prevents FastAPI JSON serialization errors
#                 bounding_boxes.append({
#                     "w": int(w), 
#                     "h": int(h), 
#                     "x": int(x), 
#                     "y": int(y)
#                 })

#     # --- NEW PRINT STATEMENT HERE ---
#     # Print the final array before passing it back to the API route
#     print(f"--- OPENCV CHECK --- Final bounding_boxes array to return: {bounding_boxes}")



#     # 3. Return the new object alongside the images
#     return cleaned_image, output_img, bounding_boxes

import os
import cv2
import numpy as np

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
    # cv2.imshow("Whitehot Crack Mask", whitehot_crack)  # Debug: Show the binary mask to verify it's correct
    # cv2.imshow("Resized Grayscale Image", resized_img)  # Debug: Show the resized grayscale image before processing
    # cv2.imshow("Cleaned Image", cleaned_image)  # Debug: Show the cleaned image to verify it's being updated correctly
    # cv2.imshow("Output Image with Boxes", output_img)  # Debug: Show the output image to verify boxes are drawn correctly

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

    # --- NEW PRINT STATEMENT HERE ---
    # Print the final array before passing it back to the API route
    print(f"--- OPENCV CHECK --- Final bounding_boxes array to return: {bounding_boxes}")

   # =========================================================================
    # --- NEWLY INSERTED CODE BELOW ---
    # =========================================================================
    
    # Create a fresh BGR representation of the resized image so we can crop 
    # clean versions without the green rectangles drawn above
    image = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    # 4. Create a copy of the image to draw on
    # We do this OUTSIDE the loop so all boxes appear on the same image
    image_with_box = image.copy()

    # --- LOOP: Process every box in the array ---
    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        
        # 5. Draw the rectangle on the main image
        start_point = (x, y)
        end_point = (x + w, y + h)
        cv2.rectangle(image_with_box, start_point, end_point, (0, 255, 0), 2)

        # 3. Crop FIRST
        cropped_roi = image[y:y+h, x:x+w]

        # 7. Display the original cropped region
        # cv2.imshow(f'Inspected Area - Box {i}', cropped_roi)

        # 8. Convert to Grayscale and display
        gray_cropped = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
        # cv2.imshow(f'Grayscale Area - Box {i}', gray_cropped)
        
        # 9. Convert to Binary Bitmap and display
        _, bitmap_cropped = cv2.threshold(gray_cropped, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow(f'Bitmap Area - Box {i}', bitmap_cropped)

        # --- PART 2: PIXEL ANALYSIS ---
        # 10. Calculate pixel totals
        total_pixels = bitmap_cropped.size
        # 11. Count white pixels
        white_pixels = cv2.countNonZero(bitmap_cropped)
        # 12. Calculate black pixels
        black_pixels = total_pixels - white_pixels
        
        if total_pixels > 0:
            #  Calculate percentages of the white and black pixels to total
            white_percentage = (white_pixels / total_pixels) * 100
            black_percentage = (black_pixels / total_pixels) * 100

            # 14. Print the results to the console
            print(f"\n--- Image Pixel Analysis (Box {i}) ---")
            print(f"Total Pixels: {total_pixels:,}")
            print(f"White Pixels (Background): {white_pixels:,} ({white_percentage:.2f}%)")
            print(f"Black Pixels (Crack):     {black_pixels:,} ({black_percentage:.2f}%)")

            # --- NEW: Save pixel data directly into the current bounding box dictionary ---
            # Casting to int() to prevent any potential JSON serialization errors in FastAPI
            box["total_pixels"] = int(total_pixels)
            box["white_pixels"] = int(white_pixels)
            box["white_percentage"] = round(white_percentage, 2)
            box["black_pixels"] = int(black_pixels)
            box["black_percentage"] = round(black_percentage, 2)

    #  Display the original image with ALL bounding boxes
    # cv2.imshow('All Bounding Boxes', image_with_box)

    # --- NEW: Print the fully updated array with pixel data right before waitKey ---
    print("\n--- FINAL BOUNDING BOXES WITH PIXEL DATA ---")
    for i, b in enumerate(bounding_boxes):
        print(f"Box {i}: {b}")

    # Wait for any key press, then close all windows 
    # (WARNING: REMOVE THIS ENTIRE BLOCK IN FASTAPI PRODUCTION)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # =========================================================================
    # --- END INSERTED CODE ---
    # =========================================================================

    # 3. Return the new object alongside the images
    return cleaned_image, output_img, bounding_boxes
