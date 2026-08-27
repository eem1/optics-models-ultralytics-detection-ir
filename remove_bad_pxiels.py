import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
input_folder = "."           # Where your raw .tif files are
output_folder = "cleaned_tifs" # Where the fixed files will go
os.makedirs(output_folder, exist_ok=True)

# Sensitivity: How much must a pixel differ from neighbors to be considered "bad"?
# For 16-bit thermal data (0-65535), a jump of 1000+ is usually an artifact.
# Lower this number if it's missing bad pixels. Raise it if it's deleting seal whiskers.
diff_threshold = 1000 

tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
print(f"Despeckling {len(tif_files)} images...")

count = 0
for tif_path in tif_files:
    # 1. Load 16-bit thermal image
    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        continue

    # 2. Create a "Reference" image using a Median Blur
    # kernel_size=3 looks at the immediate 8 neighbors
    # This "smooths" the image by ignoring outliers
    median_blurred = cv2.medianBlur(img, 3)

    # 3. Identify Bad Pixels (The "Surgical" Step)
    # Calculate absolute difference between Original and Median
    diff = cv2.absdiff(img, median_blurred)
    
    # Create a mask: True where the pixel is a "bad" outlier
    bad_pixel_mask = diff > diff_threshold

    # 4. Fix ONLY the bad pixels
    # Start with a copy of the original
    cleaned_img = img.copy()
    
    # Wherever the mask is True, replace original pixel with the median value
    cleaned_img[bad_pixel_mask] = median_blurred[bad_pixel_mask]

    # Calculate stats for the user
    num_bad_pixels = np.sum(bad_pixel_mask)
    
    if num_bad_pixels > 0:
        # Save result
        filename = os.path.basename(tif_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, cleaned_img)
        
        count += 1
        if count % 10 == 0:
            print(f"Fixed {num_bad_pixels} pixels in: {filename}")
    else:
        # Just copy the file if no changes (optional, saves disk space)
        pass

print(f"Done! Cleaned images saved to '{output_folder}'.")