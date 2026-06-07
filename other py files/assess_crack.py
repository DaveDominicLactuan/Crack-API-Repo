import os
import glob
import cv2
import numpy as np
from scipy import stats

PIXEL_TO_MM = 0.1
INPUT_FOLDER = "preprocessed_cw"
OUTPUT_FOLDER = "assessed_cracks"

def load_binary_mask(file_path, threshold=0.5):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        something, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary_mask
    elif ext in ['.txt', '.csv']:
        prob_matrix = np.loadtxt(file_path)
        return (prob_matrix >= threshold).astype(np.uint8) * 255 # typecast to 8-bit channel map
    return None

def assess_crack(binary_mask):
    contours, something = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(contours) # RETR_EXTERNAL = return outermost contours / CHAIN_APPROX_NONE = get every single point (no approximation)
    crack_records = []

    for count in contours:
        if cv2.contourArea(count) < 5:
            continue

        x, y, w, h = cv2.boundingRect(count)
        region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8) # force 8-bit integers to match 8-bit map
        shifted_count = count - [x - 5, y - 5]
        cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1) # -1 = draw every shape in list, 255 = paint white, -1 = completely fill
        dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5) # DIST_L2 = Euclidian distance / straight line, 5 = mask size
        skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL) # GUOHALL vs ZHANGSUEN to better preserve structural connectivity
        pixel_length = np.sum(skeleton == 255)
        length_mm = pixel_length * PIXEL_TO_MM

        if pixel_length == 0: continue

        widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
        max_w = np.max(widths_mm)
        mean_w = np.mean(widths_mm)
        mode_w = float(stats.mode(np.round(widths_mm, 2), keepdims=True).mode)
        rotated_box = cv2.minAreaRect(count)
        orientation_angle = rotated_box[2]

        crack_records.append({
            "contour": count,
            "rotated_box": rotated_box,
            "length_mm": length_mm,
            "max_width_mm": max_w,
            "mean_width_mm": mean_w,
            "mode_width_mm": mode_w,
            "orientation_deg": orientation_angle
        })

    return crack_records

def generate_and_save_outputs(file_path, binary_mask, records, output_dir):
    base = os.path.splitext(os.path.basename(file_path))[0]
    img_contour = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    img_bbox = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    text_log = []

    for idx, crack in enumerate(records):
        cv2.drawContours(img_contour, [crack["contour"]], -1, (0, 0, 255), 2)

        box_points = cv2.boxPoints(crack["rotated_box"])
        box_points = np.int64(box_points)
        cv2.drawContours(img_bbox, [box_points], 0, (0, 255, 0), 2)

        text_log.append(
            f"Crack_{idx} -> Length: {crack['length_mm']:.2f}mm\n"
            f"Max_W: {crack['max_width_mm']:.2f}mm\nMean_W: {crack['mean_width_mm']:.2f}mm \n"
            f"Mode_W: {crack['mode_width_mm']:.2f}mm\nOrientation: {crack['orientation_deg']:.1f} degrees\n"
        )

    cv2.imwrite(os.path.join(output_dir, f"{base}_contour.jpg"), img_contour)
    cv2.imwrite(os.path.join(output_dir, f"{base}_bbox.jpg"), img_bbox)
    with open(os.path.join(output_dir, f"{base}_metrics.txt"), "w") as f:
        f.writelines(text_log)

def run_pipeline(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    os.makedirs(output_folder, exist_ok=True)

    for path in glob.glob(os.path.join(input_folder, "*.*")):
        mask = load_binary_mask(path)
        if mask is not None:
            data = assess_crack(mask)
            generate_and_save_outputs(path, mask, data, output_folder)

if __name__ == "__main__":
    run_pipeline()