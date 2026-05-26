import os
import numpy as np
import cv2
import glob
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

def load_raw(path, height, width):
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.uint16)
    return raw.reshape(height, width).astype(np.float32)

def normalize(raw, black_level=240, white_level=4095):
    return np.clip((raw - black_level) / (white_level - black_level), 0, 1)

def load_png(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def tof(frame1, frame2):
    """temporal Optical Flow"""
    f1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

def to_tensor(img):
    """H x W x C numpy -> 1 x C x H x W tensor (-1~1)"""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t * 2 - 1

parser = argparse.ArgumentParser()
parser.add_argument('--denoised_raw_dir', type=str, default='./inference/data/output/')
parser.add_argument('--noisy_raw_dir', type=str, default='./inference/data/input/')
parser.add_argument('--gt_raw_dir', type=str, default='./inference/data/gt_raw/')
parser.add_argument('--denoised_png_dir', type=str, default='./inference/data/rgb_output/')
parser.add_argument('--noisy_png_dir', type=str, default='./inference/data/rgb_input/')
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--width', type=int, default=1920)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = lpips.LPIPS(net='alex').to(device)

# 파일 로드
denoised_raws = sorted(glob.glob(os.path.join(args.denoised_raw_dir, '*.raw')))
noisy_raws = sorted(glob.glob(os.path.join(args.noisy_raw_dir, '*.raw')))
gt_raws = sorted(glob.glob(os.path.join(args.gt_raw_dir, '*.raw')))
denoised_pngs = sorted(glob.glob(os.path.join(args.denoised_png_dir, '*.png')))
noisy_pngs = sorted(glob.glob(os.path.join(args.noisy_png_dir, '*.png')))

# 결과 저장
results = {
    'denoised_psnr': [], 'denoised_ssim': [],
    'noisy_psnr': [], 'noisy_ssim': [],
    'denoised_tlpips': [], 'noisy_tlpips': [],
    'denoised_tof': [], 'noisy_tof': [],
}

print("=== PSNR / SSIM ===")
for i, (d, n, g) in enumerate(zip(denoised_raws, noisy_raws, gt_raws)):
    den = normalize(load_raw(d, args.height, args.width))
    noi = normalize(load_raw(n, args.height, args.width))
    gt = normalize(load_raw(g, args.height, args.width))

    dp = psnr(gt, den, data_range=1.0)
    ds = ssim(gt, den, data_range=1.0)
    np_ = psnr(gt, noi, data_range=1.0)
    ns = ssim(gt, noi, data_range=1.0)

    results['denoised_psnr'].append(dp)
    results['denoised_ssim'].append(ds)
    results['noisy_psnr'].append(np_)
    results['noisy_ssim'].append(ns)
    print(f"frame {i:03d} | denoised PSNR: {dp:.4f} SSIM: {ds:.4f} | noisy PSNR: {np_:.4f} SSIM: {ns:.4f}")

print("\n=== tOF / TLPIPS ===")
for i in range(len(denoised_pngs) - 1):
    d1 = load_png(denoised_pngs[i])
    d2 = load_png(denoised_pngs[i+1])
    n1 = load_png(noisy_pngs[i])
    n2 = load_png(noisy_pngs[i+1])

    # tOF
    d_tof = tof(d1, d2)
    n_tof = tof(n1, n2)
    results['denoised_tof'].append(d_tof)
    results['noisy_tof'].append(n_tof)

    # TLPIPS
    with torch.no_grad():
        d_lpips = loss_fn(to_tensor(d1).to(device), to_tensor(d2).to(device)).item()
        n_lpips = loss_fn(to_tensor(n1).to(device), to_tensor(n2).to(device)).item()
    results['denoised_tlpips'].append(d_lpips)
    results['noisy_tlpips'].append(n_lpips)

    print(f"frame {i:03d}-{i+1:03d} | denoised tOF: {d_tof:.4f} TLPIPS: {d_lpips:.4f} | noisy tOF: {n_tof:.4f} TLPIPS: {n_lpips:.4f}")

print("\n=== 평균 결과 ===")
print(f"{'':20} {'denoised':>12} {'noisy':>12}")
print(f"{'PSNR':20} {np.mean(results['denoised_psnr']):>12.4f} {np.mean(results['noisy_psnr']):>12.4f}")
print(f"{'SSIM':20} {np.mean(results['denoised_ssim']):>12.4f} {np.mean(results['noisy_ssim']):>12.4f}")
print(f"{'tOF':20} {np.mean(results['denoised_tof']):>12.4f} {np.mean(results['noisy_tof']):>12.4f}")
print(f"{'TLPIPS':20} {np.mean(results['denoised_tlpips']):>12.4f} {np.mean(results['noisy_tlpips']):>12.4f}")