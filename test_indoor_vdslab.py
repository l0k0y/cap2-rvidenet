# import os
# import glob
# import numpy as np
# import rawpy
# import torch
# import random
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from utils import *

# # 모델 로드
# model = torch.load("/database/iyj0121/ckpt/rvidnet/rvidnet_fintune_self_record_data_scene1_9_signal_independent_dependent_neif_noise_estimation_uniform_noise_ADC_modeling_gain_6_29p4/model_epoch70.pth").cuda()
# model.eval()

# input_dir_base = "/database/iyj0121/dataset/val_vnt/scene12_purpledog/"
# gt_dir = "/database/iyj0121/dataset/val_vnt/scene12_purpledog_gt/"

# psnr_values = []
# black_level = 240
# max_value = 2**12 - 1 

# def load_raw_image(file_path):
#     raw = rawpy.imread(file_path)
#     return raw.raw_image_visible.astype(np.uint16)

# def process_frame(frame_num):
#     # scene12_frame{frame_num-1}, scene12_frame{frame_num}, scene12_frame{frame_num+1} 디렉토리에서 각 프레임의 long/dump_bayer_frame_*.raw 파일 랜덤으로 선택
#     frames_to_load = [frame_num - 1, frame_num, frame_num + 1]

#     # 범위를 벗어나는 인덱스에 대한 처리 (경계 처리)
#     frames_to_load[0] = max(frames_to_load[0], 0)  # 첫 프레임보다 작은 경우 0으로
#     frames_to_load[2] = min(frames_to_load[2], 60)  # 마지막 프레임보다 큰 경우 60으로

#     frame_list = []
    
#     for scene_idx in frames_to_load:
#         input_files = sorted(glob.glob(os.path.join(input_dir_base, f'scene12_frame{scene_idx}', 'long', '*.raw')))
        
#         # 각 scene12_frame 디렉토리 내에서 랜덤하게 dump_bayer_frame_*.raw 파일을 선택
#         if len(input_files) == 0:
#             print(f"No files found in scene12_frame{scene_idx}")
#             continue
#         random_idx = random.randint(0, 100)
#         random_file_name = f'dump_bayer_frame_{random_idx:05d}.raw'
#         random_file_path = os.path.join(input_dir_base, f'scene12_frame{scene_idx}', 'long', random_file_name)
#         raw_image = load_raw_image(random_file_path)
#         print(f"Loaded frame {scene_idx} from {random_file_path}")
#         input_full = np.expand_dims(pack_rggb_raw(raw_image), axis=0)
#         frame_list.append(input_full)

#     if len(frame_list) < 3:
#         print(f"Not enough frames to process scene {frame_num}")
#         return

#     # 선택한 3개의 프레임을 concatenate (시간축으로)
#     input_data = np.concatenate(frame_list, axis=3)

#     # 모델을 사용하여 디노이징 수행
#     denoised_image = test_big_size_raw(input_data, model, patch_h=256, patch_w=256, patch_h_overlap=64, patch_w_overlap=64)
#     denoised_image = depack_rggb_raw(denoised_image.squeeze())

#     # GT 이미지 로드 (현재 frame_num에 해당하는 GT 이미지)
#     gt_image_path = os.path.join(gt_dir, f'average_image_{frame_num:03d}.raw')
#     gt_image = np.fromfile(gt_image_path, dtype=np.uint16).reshape(denoised_image.shape)

#     # Black level 제거 및 정규화
#     gt_image_normalized = (gt_image.astype(np.float32) - black_level) / (max_value - black_level)
#     denoised_image_normalized = (denoised_image.astype(np.float32) - black_level) / (max_value - black_level)

#     # PSNR 계산
#     psnr_value = compare_psnr(gt_image_normalized, denoised_image_normalized, data_range=1.0)
#     psnr_values.append(psnr_value)
#     print(f"Frame {frame_num}: PSNR calculated, PSNR: {psnr_value}")

#     # 결과 저장
#     save_result = (denoised_image * ((2**12 - 1) - 240) + 240).astype(np.uint16)
#     save_path = os.path.join("/database/iyj0121/result/rvidenet_val/rvidnet_fintune_self_record_data_scene1_9_signal_independent_dependent_neif_noise_estimation_uniform_noise_ADC_modeling_gain_6_29p4/", f'denoised_raw_frame_{frame_num:03d}.raw')
    
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, 'wb') as f:
#         f.write(save_result.tobytes())

# # frame0부터 frame60까지 모든 프레임 처리
# for frame_num in range(61):
#     process_frame(frame_num)

# # PSNR 평균 출력
# avg_psnr = np.mean(psnr_values)
# print(f"Average PSNR: {avg_psnr}")

import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils import compute_ssim_for_packed_raw

denoised_path = "/database/iyj0121/result/tcsvt_final/tcsvt_final_inference_result/baseline_rvidenet_scene12_ablation2/denoised_raw_frame_{:06d}.raw"
#denoised_path = "/database/iyj0121/dataset/val_vnt/scene12/dump_bayer_frame_{:05d}.raw"
#denoised_path = "/database/iyj0121/dataset/val_vnt/scene12/dump_bayer_frame_{:05d}.raw"
gt_path = "/database/iyj0121/result/tcsvt_final/tcsvt_final_val_dataset/scene12_purpledog_gt/average_image_{:03d}.raw"

frames_to_process = range(61)

total_psnr = 0

psnr_values = []
ssim_values = []

black_level = 240
max_value = 2**12 - 1  

def read_raw_image(file_path, height=1080, width=1920):
    raw_data = np.fromfile(file_path, dtype=np.uint16)
    if raw_data.size != height * width:
        raise ValueError(f"파일 크기가 예상과 다릅니다: {file_path}")
    return raw_data.reshape((height, width))

for i in frames_to_process:
    denoised_file = denoised_path.format(i)
    gt_file = gt_path.format(i)

    if not os.path.exists(denoised_file):
        print(f"File not found: {denoised_file}")
        continue
    if not os.path.exists(gt_file):
        print(f"File not found: {gt_file}")
        continue

    denoised_image = read_raw_image(denoised_file)
    gt_image = read_raw_image(gt_file)

    denoised_image_normalized = (denoised_image.astype(np.float32) - black_level) / (max_value - black_level)
    gt_image_normalized = (gt_image.astype(np.float32) - black_level) / (max_value - black_level)

    psnr_value = compare_psnr(gt_image_normalized, denoised_image_normalized, data_range=1.0)
    ssim_value = compute_ssim_for_packed_raw(gt_image_normalized, denoised_image_normalized)
    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    total_psnr += psnr_value
    
    print(f"Frame {i}: PSNR = {psnr_value}")
    print(f"Frame {i}: SSIM = {ssim_value}")

if len(psnr_values) > 0:
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)
    print(f"Average PSNR over all frames: {average_psnr}")
    print(f"Average SSIM over all frames: {average_ssim}")
else:
    print("No PSNR values were calculated.")

total_psnr /= len(psnr_values)

print(f"Total PSNR: {total_psnr}")
