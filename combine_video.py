import cv2
import os
import glob
from tqdm import tqdm

def images_to_video(image_folder, output_path, fps=30):
    """
    将图片序列转换为视频
    
    Args:
        image_folder: 图片文件夹路径
        output_path: 输出视频路径
        fps: 视频帧率
    """
    # 获取所有图片文件
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    # 读取第一张图片获取尺寸
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 逐帧写入视频
    print(f"Converting images in {image_folder} to video...")
    for image_path in tqdm(images):
        frame = cv2.imread(image_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

def main():
    # 基础路径
    base_path = "/Users/iamzlt/Desktop/姿态估计项目/MultiPose/Campus_Seq1/frames"
    
    # 处理每个摄像头的图片序列
    for camera in ["Camera0", "Camera1", "Camera2"]:
        image_folder = os.path.join(base_path, camera)
        output_path = os.path.join(base_path, f"{camera}.mp4")
        
        if os.path.exists(image_folder):
            images_to_video(image_folder, output_path)
        else:
            print(f"Folder not found: {image_folder}")

if __name__ == "__main__":
    main()