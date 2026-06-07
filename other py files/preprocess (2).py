import os
import glob
import cv2
import numpy as np

def preprocess_and_save_format(image_path):
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

    # final_array = whitehot_crack.astype(np.float32) / 255.0
    # final_tensor = np.expand_dims(final_array, axis=-1)

    num_labels, labels, stats, centriods = cv2.connectedComponentsWithStats(whitehot_crack, connectivity=8, ltype=cv2.CV_32S)
    cleaned_image = np.zeros_like(whitehot_crack)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 16:
            cleaned_image[labels == i] = 255

    return cleaned_image
    # smoothed = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced_img = clahe.apply(smoothed)
    # return enhanced_img # Keep it as 0-255 integers so OpenCV can save it as a picture

input_folder = "CW"
output_folder = "preprocessed_cw"

os.makedirs(output_folder, exist_ok=True)

image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

print(f"Saving {len(image_paths)} processed copies to: {output_folder}")

for path in image_paths:
    processed_img = preprocess_and_save_format(path)
    
    file_name = os.path.basename(path)
    
    save_path = os.path.join(output_folder, file_name)
    
    cv2.imwrite(save_path, processed_img)

print("Done! All copies saved.")


# TRY USING THIS FOR PREPROCESS BOUNDING BOX TO FILE TO PAIR WITH IMAGE FOR TRAINING
# for i in range(1, num_labels):
#     # Extract structural properties
#     x = stats[i, cv2.CC_STAT_LEFT]
#     y = stats[i, cv2.CC_STAT_TOP]
#     w = stats[i, cv2.CC_STAT_WIDTH]
#     h = stats[i, cv2.CC_STAT_HEIGHT]
#     area = stats[i, cv2.CC_STAT_AREA]
    
#     # Get the center point coordinates
#     cx, cy = centroids[i]
    
#     # Draw green bounding box around foreground components
#     cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     # Mark the center with a small red circle
#     cv2.circle(output_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    
#     # Output metrics to console
#     print(f"Component {i}: Area = {area}px, Center = ({cx:.1f}, {cy:.1f})")