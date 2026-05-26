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
from video_denoising_models import FastDVDnet, FastDVDnetHR
#from fastdvdnet import FastDVDnet


parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--model', dest='model', type=str, default='finetune', help='model type')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--input_dir', type=str, default="/database/iyj0121/result/tcsvt_final/tcsvt_final_val_dataset/scene5_noisy_video/", help='input path') #   "/database/iyj0121/dataset/hanhwa/raw_0317/"
parser.add_argument('--output_dir', type=str, default="/database/iyj0121/result/tcsvt_final/tcsvt_final_inference_result/starlight_hrnet_scene5/", help='output path')
parser.add_argument('--rgb_output_dir', type=str, default="/database/iyj0121/result/rvidnet_self_record_data_scene1_7_rgb/", help='output path')
parser.add_argument('--vis_data', type=bool, default=False, help='whether to visualize noisy and gt data')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
model = FastDVDnet(num_input_frames=5).cuda()
#model = torch.load("/database/iyj0121/ckpt/rvidenet/finetune_signal_independent_dependent_neif_noise_estimation_gain_only_29p4_FastDVDnet_no_predenoising_5frame/model_epoch70.pth").cuda()
state_dict = torch.load("/database/iyj0121/ckpt/starlight/checkpoint450_test_loss0.04863_trainloss0.1003.pt")
if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)


input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.raw')))
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.isdir(args.rgb_output_dir):
    os.makedirs(args.rgb_output_dir)

for frame_idx, input_file in enumerate(input_files):
    print('processing frame {}'.format(frame_idx))

    frame_list = []
    for j in range(-2, 3):
        idx = frame_idx + j
        if idx < 0:
            idx = 0
        elif idx >= len(input_files):
            idx = len(input_files) - 1
        raw = rawpy.imread(input_files[idx]).raw_image_visible.astype(np.uint16)
        input_full = np.expand_dims(pack_rggb_raw(raw), axis=0)
        frame_list.append(input_full)

    input_data = np.concatenate(frame_list, axis=3)
    #input_data = input_data.transpose(0, 3, 1, 2)
    #print("input_data shape:", input_data.shape)
    test_result = test_big_size_raw_5frame(input_data, model, patch_h=256, patch_w=256, patch_h_overlap=64, patch_w_overlap=64)
    test_result = test_result.squeeze()
    test_result = depack_rggb_raw(test_result)
    save_result = (test_result * ((2**12-1)- 240) + 240).astype(np.uint16)
    save_pass = os.path.join(args.output_dir, 'denoised_raw_frame_{:06d}.raw'.format(frame_idx))
    with open(save_pass, 'wb') as f:
        f.write(save_result.tobytes())
        f.close()

    noisy_raw_frame = preprocess(input_data[:, :, :, 8:12])
    noisy_srgb_frame = postprocess(isp(noisy_raw_frame))[0]
    if args.vis_data:
        cv2.imwrite(os.path.join(args.output_dir, 'frame_{:06d}_noisy_sRGB.png'.format(frame_idx)), np.uint8(noisy_srgb_frame * 255))

    denoised_raw_frame = preprocess(np.expand_dims(pack_rggb_raw(test_result), axis=0))
    denoised_srgb_frame = postprocess(isp(denoised_raw_frame))[0]
    denoised_srgb_frame = np.power(denoised_srgb_frame, 1/2.2)
    print("denoised_srgb_frame min/max:", denoised_srgb_frame.min(), denoised_srgb_frame.max())
    scaled_frame = np.uint8(denoised_srgb_frame * 255)
    print("scaled_frame min/max:", scaled_frame.min(), scaled_frame.max())
    #cv2.imwrite(os.path.join(args.rgb_output_dir, 'frame_{:06d}_denoised_sRGB.png'.format(frame_idx)), np.uint8(denoised_srgb_frame * 255), cv2.COLOR_RGB2BGR)
    #bgr_frame = cv2.cvtColor(scaled_frame, cv2.COL)
    cv2.imwrite(os.path.join(args.rgb_output_dir, 'frame_{:06d}_denoised_sRGB.png'.format(frame_idx)), scaled_frame)
