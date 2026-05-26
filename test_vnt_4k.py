from __future__ import division
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from utils import *
import rawpy

def depack_rggb_raw(packed_raw):
    H, W, _ = packed_raw.shape
    output = np.zeros((H * 2, W * 2), dtype=packed_raw.dtype)

    output[0:H*2:2, 0:W*2:2] = packed_raw[:, :, 0]  # R
    output[0:H*2:2, 1:W*2:2] = packed_raw[:, :, 1]  # G1
    output[1:H*2:2, 1:W*2:2] = packed_raw[:, :, 2]  # G2
    output[1:H*2:2, 0:W*2:2] = packed_raw[:, :, 3]  # B

    return output

def pack_rggb_raw_(raw):
    #pack RGGB Bayer raw to 4 channels
    black_level = 256
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--model', dest='model', type=str, default='finetune', help='model type')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--input_dir', type=str, default="/database/iyj0121/dataset/4k_vnt/ckpt/finetune/samsung_s4_ev0_rggb/", help='input path') #     "/database/iyj0121/dataset/hanhwa/raw_0317/"
parser.add_argument('--output_dir', type=str, default="/database/iyj0121/dataset/4k_vnt/ckpt/finetune/samsung_s4_ev0_rggb_gain_6_30_hcg_scene_from_1_to_6_video_model_size_up/", help='output path')
parser.add_argument('--rgb_output_dir', type=str, default="/database/iyj0121/result/rvidnet_self_record_data_scene1_7_rgb/", help='output path')
parser.add_argument('--vis_data', type=bool, default=False, help='whether to visualize noisy and gt data')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
model = torch.load("/database/iyj0121/dataset/4k_vnt/ckpt/finetune/finetune_noise_estimation_crvd_method_neif_on_4k_calibration_gain_6_30_hcg_scene_from_1_to_6_video/model_epoch500.pth").cuda()

input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.raw')))
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.isdir(args.rgb_output_dir):
    os.makedirs(args.rgb_output_dir)

for frame_idx, input_file in enumerate(input_files):
    print('processing frame {}'.format(frame_idx))

    frame_list = []
    for j in range(-1, 2):
        idx = frame_idx + j
        if idx < 0:
            idx = 0
        elif idx >= len(input_files):
            idx = len(input_files) - 1

        with open(input_files[idx], 'rb') as f:
            raw = np.fromfile(f, dtype=np.uint16)
            #raw = raw.reshape(512, 512)  
            raw = raw.reshape(3000, 4000)
        #raw = rawpy.imread(input_files[idx]).raw_image_visible.astype(np.uint16)
        input_full = np.expand_dims(pack_rggb_raw_(raw), axis=0)
        frame_list.append(input_full)

    input_data = np.concatenate(frame_list, axis=3)
    test_result = test_big_size_raw(input_data, model, patch_h=256, patch_w=256, patch_h_overlap=64, patch_w_overlap=64)
    test_result = test_result.squeeze()
    test_result = depack_rggb_raw(test_result)
    save_result = (test_result * ((2**12-1)- 256) + 256).astype(np.uint16)
    save_pass = os.path.join(args.output_dir, 'denoised_raw_frame_{:06d}.raw'.format(frame_idx))
    with open(save_pass, 'wb') as f:
        f.write(save_result.tobytes())
        f.close()

    noisy_raw_frame = preprocess(input_data[:, :, :, 4:8])
    noisy_srgb_frame = postprocess(isp(noisy_raw_frame))[0]
    if args.vis_data:
        cv2.imwrite(os.path.join(args.output_dir, 'frame_{:06d}_noisy_sRGB.png'.format(frame_idx)), np.uint8(noisy_srgb_frame * 255))

    denoised_raw_frame = preprocess(np.expand_dims(pack_rggb_raw_(test_result), axis=0))
    denoised_srgb_frame = postprocess(isp(denoised_raw_frame))[0]
    denoised_srgb_frame = np.power(denoised_srgb_frame, 1/2.2)
    print("denoised_srgb_frame min/max:", denoised_srgb_frame.min(), denoised_srgb_frame.max())
    scaled_frame = np.uint8(denoised_srgb_frame * 255)
    print("scaled_frame min/max:", scaled_frame.min(), scaled_frame.max())
    #cv2.imwrite(os.path.join(args.rgb_output_dir, 'frame_{:06d}_denoised_sRGB.png'.format(frame_idx)), np.uint8(denoised_srgb_frame * 255), cv2.COLOR_RGB2BGR)
    #bgr_frame = cv2.cvtColor(scaled_frame, cv2.COL)
    cv2.imwrite(os.path.join(args.rgb_output_dir, 'frame_{:06d}_denoised_sRGB.png'.format(frame_idx)), scaled_frame)
