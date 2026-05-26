import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.stats import poisson
import rawpy

def generate_noisy_raw(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw - 240) / a).rvs() * a
    gaussian_noise = np.sqrt(gaussian_noise_var) * np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img, 0), 2**12 - 1)
    
    return noisy_img

iso = 25600
a = 52.032536
b = 1819.818657

data_id = '02'
raw_paths = [f'/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-{data_id}_raw/{i:06d}.raw' for i in range(1, 601)]
save_path = f'/database/iyj0121/dataset/SRVD_data/raw_noisy/MOT17-{data_id}_raw/'

if not os.path.isdir(save_path):
    os.makedirs(save_path)

for raw_path in raw_paths:
    clean_raw = rawpy.imread(raw_path).raw_image.astype(np.float32)
    #clean_raw = cv2.imread(raw_path, -1)
    #clean_raw = clean_raw.astype(np.float32)
    
    noisy_raw = generate_noisy_raw(clean_raw, a, b)
    noisy_save = Image.fromarray(np.uint16(noisy_raw))
    base_name = os.path.basename(raw_path)[:-4]
    with open((os.path.join(save_path, base_name + '.raw')), 'wb') as f:
        f.write(noisy_save.tobytes())
        f.close()
    print('have synthesized noise on MOT17-{}_raw '.format(data_id) + base_name + '.raw')