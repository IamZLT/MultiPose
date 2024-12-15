import cv2

def compress_video(input_path, output_path, target_size_mb=10):
    # 读取输入视频
    cap = cv2.VideoCapture(input_path)
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算目标比特率
    target_size_bytes = target_size_mb * 1024 * 1024
    duration = frame_count / fps
    target_bitrate = int(target_size_bytes * 8 / duration)
    
    # 设置输出视频的编码器和参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )
    
    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 可以在这里添加额外的帧处理，比如调整分辨率
        # frame = cv2.resize(frame, (new_width, new_height))
        
        out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"Video compressed and saved to {output_path}")

if __name__ == "__main__":
    input_video = "assert/videos/3d_pose.mp4"
    output_video = "assert/videos/3d_pose_compressed.mp4"
    target_size_mb = 10  # 目标文件大小（MB）
    
    compress_video(input_video, output_video, target_size_mb) 