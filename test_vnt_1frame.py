from __future__ import division
import os
import torch
import numpy as np
import glob
import cv2
import argparse
from utils import *
import rawpy

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--model', dest='model', type=str, default='finetune', help='model type')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--input_dir', type=str, default="/database/iyj0121/dataset/4k_vnt/ckpt/finetune/samsung_noise_frame_rggb/", help='input path') 
parser.add_argument('--output_dir', type=str, default="/database/iyj0121/dataset/4k_vnt/ckpt/finetune/samsung_denoise_frame_rggb_s4_ev6/", help='output path')
parser.add_argument('--rgb_output_dir', type=str, default="/database/iyj0121/result/rvidnet_self_record_data_scene1_7_rgb/", help='output path')
parser.add_argument('--vis_data', type=bool, default=False, help='whether to visualize noisy and gt data')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
model = torch.load("/database/iyj0121/ckpt/lightweight/unet_pretrain_mot_pack_debug/model_epoch3000.pth").cuda()

input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.raw')))
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.isdir(args.rgb_output_dir):
    os.makedirs(args.rgb_output_dir)

for frame_idx, input_file in enumerate(input_files):
    print('processing frame {}'.format(frame_idx))
    
    # 하나의 frame만 처리하도록 변경
    raw = rawpy.imread(input_file).raw_image_visible.astype(np.uint16)
    input_data = np.expand_dims(pack_rggb_raw(raw), axis=0)

    test_result = test_big_size_raw_1frame(input_data, model, patch_h=256, patch_w=256, patch_h_overlap=64, patch_w_overlap=64)
    test_result = test_result.squeeze()
    test_result = depack_rggb_raw(test_result)
    save_result = (test_result * ((2**12 - 1) - 256) + 256).astype(np.uint16)
    save_pass = os.path.join(args.output_dir, 'denoised_raw_frame_{:06d}.raw'.format(frame_idx))
    with open(save_pass, 'wb') as f:
        f.write(save_result.tobytes())

    # 시각화를 위한 처리
    noisy_raw_frame = preprocess(input_data)
    noisy_srgb_frame = postprocess(isp(noisy_raw_frame))[0]
    if args.vis_data:
        cv2.imwrite(os.path.join(args.output_dir, 'frame_{:06d}_noisy_sRGB.png'.format(frame_idx)), np.uint8(noisy_srgb_frame * 255))

    denoised_raw_frame = preprocess(np.expand_dims(pack_rggb_raw(test_result), axis=0))
    denoised_srgb_frame = postprocess(isp(denoised_raw_frame))[0]
    denoised_srgb_frame = np.power(denoised_srgb_frame, 1 / 2.2)
    print("denoised_srgb_frame min/max:", denoised_srgb_frame.min(), denoised_srgb_frame.max())
    scaled_frame = np.uint8(denoised_srgb_frame * 255)
    print("scaled_frame min/max:", scaled_frame.min(), scaled_frame.max())
    cv2.imwrite(os.path.join(args.rgb_output_dir, 'frame_{:06d}_denoised_sRGB.png'.format(frame_idx)), scaled_frame)
