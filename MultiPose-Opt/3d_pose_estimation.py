import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import matplotlib  
matplotlib.use('Agg')  # 使用非交互式后端  
import torch
from core.read_camera_params import read_camera_params, visualize_cameras, get_projection_matrix
from core.pose_estimator import triangulate_point_from_multiple_views_linear_torch, triangulate_points_with_confidence
from core.keypoint_kalman_filter import KeypointKalmanFilter
from core.skeleton_constraint import SkeletonConstraint
from visualization.pose_visualizer import visualize_frame

def load_keypoints(camera_a_dir, camera_b_dir):  
    """加载两个相机的关键点数据和置信度"""  
    camera_a_files = sorted(os.listdir(camera_a_dir))  
    camera_b_files = sorted(os.listdir(camera_b_dir))  
    
    if len(camera_a_files) != len(camera_b_files):  
        raise ValueError("两个相机的帧数不相等")  
    
    keypoints_a = []  
    keypoints_b = []  
    scores_a = []    # 存储相机A的置信度  
    scores_b = []    # 存储相机B的置信度  
    
    for file_a, file_b in zip(camera_a_files, camera_b_files):  
        # 加载相机A的关键点和置信度  
        with open(os.path.join(camera_a_dir, file_a), 'r') as f:  
            data_a = json.load(f)  
            # 只取前17个关���点  
            points_a = np.array([shape['points'][0] for shape in data_a['shapes'][:17]],   
                              dtype=np.float32)  
            # 获取置信度  
            score_a = np.array([float(shape['score'][0]) for shape in data_a['shapes'][:17]],   
                             dtype=np.float32)  
        keypoints_a.append(points_a)  
        scores_a.append(score_a)  
        
        # 加载相机B的关键点和置信度  
        with open(os.path.join(camera_b_dir, file_b), 'r') as f:  
            data_b = json.load(f)  
            # 只取前17个关键点  
            points_b = np.array([shape['points'][0] for shape in data_b['shapes'][:17]],   
                              dtype=np.float32)  
            # 获取置信度  
            score_b = np.array([float(shape['score'][0]) for shape in data_b['shapes'][:17]],   
                             dtype=np.float32)  
        keypoints_b.append(points_b)  
        scores_b.append(score_b)  
    
    return (np.array(keypoints_a), np.array(keypoints_b),   
            np.array(scores_a), np.array(scores_b))





def calculate_height(points_3d):  
    """计算3D点云中的身高（仅考虑垂直方向）  
    
    Args:  
        points_3d: shape为(N, 3)的numpy数组，表示3D关键点坐标  
        
    Returns:  
        float: 身高(米)  
    """  
    # 获取头部(点0)和双脚(点15,16)的坐标  
    head = points_3d[0]  
    left_foot = points_3d[15]  
    right_foot = points_3d[16]  
    
    # 计算双脚中点  
    feet_center = (left_foot + right_foot) / 2  
    
    # 只取垂直方向的距离（假设y轴是垂直方向，如果是z轴请改为[2]）  
    height = abs(head[2] - feet_center[2])  
    
    return height  

def main():
    # 读取相机参数
    camera_params = read_camera_params("/home/zlt/Code/MultiPose-Tools/data/ours/params.parquet")
    
    # 加载关键点数据
    keypoints_a, keypoints_b, scores_a, scores_b = load_keypoints(  
        "/home/zlt/Code/MultiPose-Tools/data/ours/camera_a/keypoint",
        "/home/zlt/Code/MultiPose-Tools/data/ours/camera_b/keypoint"
    )
    
    # 获取投影矩阵
    P1 = get_projection_matrix(camera_params.iloc[0])
    P2 = get_projection_matrix(camera_params.iloc[1])
    
    # 初始化骨架约束
    skeleton_constraint = SkeletonConstraint()
    
    # 直接处理所有帧，不需要单独的校准阶段
    print("\n处理3D重建和骨架约束...")
    constrained_frames = []
    for i in range(len(keypoints_a)):
        # 三角测量
        points_3d = triangulate_points_with_confidence(
            keypoints_a[i], 
            keypoints_b[i], 
            P1, 
            P2,
            scores_a[i],
            scores_b[i]
        )
        
        # 计算综合置信度
        confidences = (scores_a[i] + scores_b[i]) / 2
        
        # 应用骨架约束（现在包含动态更新）
        constrained_points_3d = skeleton_constraint.optimize_points(
            points_3d, 
            confidences
        )
        
        constrained_frames.append(constrained_points_3d)
        
        # 检查骨骼长度违反情况并输出警告
        violations = skeleton_constraint.check_length_violations(constrained_points_3d)
        if violations:
            print(f"\nFrame {i} 骨骼长度违反警告:")
            for v in violations:
                print(f"连接 {v['connection']}: "
                      f"当前长度 {v['current_length']:.3f}, "
                      f"目标范围 [{v['min_length']:.3f}, {v['max_length']:.3f}], "
                      f"误差率 {v['error_ratio']:.1%}")
    
    # 第二阶段：应用卡尔曼滤波进行时间序列平滑
    print("\n应用卡尔曼滤波平滑...")
    kalman_filter = KeypointKalmanFilter()
    
    # 将所有帧转换为numpy数组以便处理
    constrained_frames = np.array(constrained_frames)
    
    # 对每个关键点进行时间序列平滑
    smoothed_frames = []
    for i in range(len(constrained_frames)):
        if i == 0:
            smoothed_frames.append(constrained_frames[i])
            continue
            
        # 使用前一帧的置信度作为参考
        confidences = (scores_a[i] + scores_b[i]) / 2
        
        # 应用卡尔曼滤波
        smoothed_points = kalman_filter.update(constrained_frames[i], confidences)
        smoothed_frames.append(smoothed_points)
    
    # 可视化处理后的结果
    print("\n生成可视化结果...")
    for i, smoothed_points in enumerate(smoothed_frames):
        visualize_frame(smoothed_points, {
            'Camera A': camera_params.iloc[0],
            'Camera B': camera_params.iloc[1]
        }, i)
        
        print(f"Processed frame {i+1}/{len(keypoints_a)}")
        # 在可视化之后添加身高计算  

    print("\n计算身高...")  
    heights = []  
    for frame_points in smoothed_frames:  
        height = calculate_height(frame_points)  
        heights.append(height)  
    
    # 计算平均身高和标准差  
    mean_height = np.mean(heights)  
    std_height = np.std(heights)  
    
    print(f"\n身高统计:")  
    print(f"平均身高: {mean_height:.2f}米")  
    print(f"标准差: {std_height:.2f}米")  
    print(f"最小身高: {min(heights):.2f}米")  
    print(f"最大身高: {max(heights):.2f}米")  


    # 创建身高分布直方图  
    plt.figure(figsize=(10, 6))  
    plt.hist(heights, bins=30, edgecolor='black')  
    plt.title('身高分布直方图')  
    plt.xlabel('身高 (米)')  
    plt.ylabel('频次')  

    # 添加平均值和标准差的垂直线  
    plt.axvline(mean_height, color='r', linestyle='dashed', linewidth=2, label=f'平均值: {mean_height:.2f}m')  
    plt.axvline(mean_height + std_height, color='g', linestyle=':', linewidth=2, label=f'+1 std: {(mean_height + std_height):.2f}m')  
    plt.axvline(mean_height - std_height, color='g', linestyle=':', linewidth=2, label=f'-1 std: {(mean_height - std_height):.2f}m')  

    plt.legend()  
    plt.grid(True, alpha=0.3)  

    # 确保output文件夹存在  
    os.makedirs('output', exist_ok=True)  

    # 保存图片  
    plt.savefig('output/height_distribution.png', dpi=300, bbox_inches='tight')  
    plt.close()  

if __name__ == "__main__":
    main() 