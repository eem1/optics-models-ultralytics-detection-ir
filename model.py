import cv2
import numpy as np
import os
import csv
import tempfile
import time
from ultralytics import YOLO
import util

# Model & Normalization Params
CONFIDENCE_THRESHOLD = 0.01
DIFF_THRESHOLD = 500
LOWER_PERCENTILE = 0.001
UPPER_PERCENTILE = 99.999

# ==========================================
# --- IMAGE PROCESSING FUNCTIONS ---
# ==========================================
def despeckle_16bit(image, threshold):
    median_blurred = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, median_blurred)
    bad_pixel_mask = diff > threshold
    cleaned_img = image.copy()
    cleaned_img[bad_pixel_mask] = median_blurred[bad_pixel_mask]
    return cleaned_img

def robust_normalize_to_8bit(image, low_p, high_p):
    p_low = np.nanpercentile(image, low_p)
    p_high = np.nanpercentile(image, high_p)
    img_clipped = np.clip(image, p_low, p_high)
    img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-5)
    img_8bit = (img_norm * 255).astype(np.uint8)
    
    # YOLO models natively expect 3 channels (RGB/BGR). 
    # We convert the 1-channel grayscale to a 3-channel image for inference compatibility.
    img_8bit_3c = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    return img_8bit_3c

def is_tif_file(filename):
    valid_exts = ( '.tif', '.tiff')
    return filename.lower().strip().endswith(valid_exts)

def run_inference(input_dir: str, output_file_path: str, config: dict):
    """
    Core inference logic for Ultralytics YOLO models.
    """
    print("[MODEL] Starting YOLO inference process...")
    print(f"[MODEL] config:{config}", flush=True)
    
    # 1. Resolve Weights
    weights_path = "/workspace/model.pt" # Default baked-in weights
    custom_weights_uri = config.get("weights")
    print(f'[MODEL] custom_weights_uri:{custom_weights_uri}', flush=True)
    
    temp_dir = tempfile.TemporaryDirectory()

    if custom_weights_uri:
        custom_weights = custom_weights_uri.split("/")[-1]
        weights_path = os.path.join(temp_dir.name, custom_weights)
        util.download_gcs_uri(custom_weights_uri, weights_path)
    
    print(f"[MODEL] Loading YOLO model: {weights_path}", flush=True)
    model = YOLO(weights_path)

    conf_threshold = config.get("options", {}).get("conf", CONFIDENCE_THRESHOLD)  
    diff_threshold = config.get("diff_threshold", DIFF_THRESHOLD)
    lower_percentitle = config.get("lower_percentitle", LOWER_PERCENTILE)
    upper_percentitle = config.get("upper_percentitle", UPPER_PERCENTILE)
    print(f"[CONFIG] confidence_threshold ={conf_threshold}, diff_threshold={diff_threshold}, lower_percentitle={lower_percentitle}, upper_percentitle={upper_percentitle}", flush=True)
    
    # 2. Discover Files with full paths
    tif_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and is_tif_file(f)
    ]
    
    if not tif_files:
        print("[MODEL] WARNING: No .tif files found in directory!")
        
    # 3. Setup CSV Output
    print(f'[MODEL] Writing output to: {output_file_path}', flush=True)
    with open(output_file_path, mode='w', newline='') as f:
        f.write("# 1: Detection or Track-id, 2: Video or Image Identifier, 3: Unique Frame Identifier, 4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y), 8: Detection or Length Confidence, 9: Target Length (0 or -1 if invalid), 10-11+: Repeated Species, Confidence Pairs or Attributes\n")
        current_time = time.ctime()
        f.write(f"# metadata,exec_time: 0,exported_by: python_yolo_script,exported_at: {current_time},,,,,,,,\n")
        writer = csv.writer(f)

        global_detection_id = 1
        total_images_processed = 0

        # 4. Process Images in Folder
        for i, tif_path in enumerate(tif_files):
            try:
                # Load image
                img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("cv2.imread returned None (corrupted or unreadable file).")

                # Normalize in-memory
                clean_16bit = despeckle_16bit(img, DIFF_THRESHOLD)
                final_8bit_3c = robust_normalize_to_8bit(clean_16bit, LOWER_PERCENTILE, UPPER_PERCENTILE)

                # Run YOLO Inference directly on the numpy array
                # save=False ensures no drawn images are saved to disk
                # verbose=False stops it from printing inference times for every single image
                results = model.predict(source=final_8bit_3c, save=False, conf=conf_threshold, verbose=False)
                
                # Because we passed a single image, results is a list of length 1
                result = results[0] 
                filename = os.path.basename(tif_path)

                # Extract detections
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                    writer.writerow([
                        global_detection_id,
                        filename,               
                        i,                      # Frame ID (using loop index)
                        f"{x_min:.3f}",         
                        f"{y_min:.3f}",         
                        f"{x_max:.3f}",         
                        f"{y_max:.3f}",         
                        f"{conf:.5f}",          
                        0,                      
                        cls_name,               
                        f"{conf:.5f}"           
                    ])
                    global_detection_id += 1

                total_images_processed += 1

            except Exception as e:
                print(f"[MODEL] Error: Failed processing image {tif_path} - ERROR: {e}", flush=True)

        print(f"[MODEL] Pipeline complete! Successfully processed {total_images_processed} images.")
        print(f"[MODEL] Results saved to: {os.path.abspath(output_file_path)}", flush=True)
        
    # Cleanup temp directory holding custom weights
    temp_dir.cleanup()
    print("[MODEL] Inference complete!", flush=True)