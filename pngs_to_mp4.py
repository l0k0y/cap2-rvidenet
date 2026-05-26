import cv2
import glob
import os

def png_to_mp4(input_dir, output_path, fps=5):
    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if not files:
        print(f"파일 없음: {input_dir}")
        return
    
    frame = cv2.imread(files[0])
    h, w = frame.shape[:2]
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for f in files:
        img = cv2.imread(f)
        writer.write(img)
    
    writer.release()
    print(f"저장 완료: {output_path}")

png_to_mp4(
    "/workspace/capstone2/inference/data/rgb_input",
    "/workspace/capstone2/inference/data/noisy.mp4"
)

png_to_mp4(
    "/workspace/capstone2/inference/data/rgb_output",
    "/workspace/capstone2/inference/data/denoised.mp4"
)