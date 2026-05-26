import rawpy
import numpy as np
import glob
import os

rawpaths = glob.glob('/database/iyj0121/dataset/ELRVD/SID/*.ARW')
save_dir = "/database/iyj0121/dataset/ELRVD/SID/long_raw_arw/"
os.makedirs(save_dir, exist_ok=True)
rawpaths.sort()

for i in range(len(rawpaths)):
    raw = rawpy.imread(rawpaths[i])
    save_path = os.path.join(save_dir, f'SID_{i}_clean.raw')
    
    img = raw.raw_image_visible
    print(img.shape)
    img = img.astype(np.float32)
    img = (img - 512) / (16383 - 512) * (4095 - 240) + 240
    img = np.clip(img, 240, 4095)
    img = img.astype(np.uint16)
    
    # img 데이터를 바이너리 형식으로 저장
    with open(save_path, 'wb') as f:
        f.write(img.tobytes())
    
    print(f"Saved {save_path}")

print("All files processed and saved.")
