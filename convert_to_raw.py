import rawpy
import numpy as np
import glob
import os

rawpaths = glob.glob('/database/iyj0121/dataset/Sony/short/*.ARW')
save_dir = "/database/iyj0121/dataset/Sony/short_raw_arw_SID/"
os.makedirs(save_dir, exist_ok=True)
rawpaths.sort()

for rawpath in rawpaths:
    raw = rawpy.imread(rawpath)
    filename = os.path.basename(rawpath)  
    save_filename = os.path.splitext(filename)[0] + '.raw' 
    save_path = os.path.join(save_dir, save_filename)
    
    img = raw.raw_image_visible
    print(img.shape)
    img = img.astype(np.float32)
    img = (img - 512) / (16383 - 512) * (4095 - 240) + 240
    img = np.clip(img, 240, 4095)
    img = img.astype(np.uint16)
    
    with open(save_path, 'wb') as f:
        f.write(img.tobytes())
    
    print(f"Saved {save_path}")

print("All files processed and saved.")
