import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import matplotlib  


class VisualizationConfig:
    def __init__(self):
        self.axis_limits = None  # 存储坐标轴范围
        
viz_config = VisualizationConfig()


def visualize_frame(frame_points_3d, camera_params, frame_idx):
    """可视化一帧的3D关键点和相机位置"""
    global viz_config
    
    # 准备相机可视化数据
    positions = []
    directions = []
    names = []
    
    # 获取相机位置和方向
    for name, params in camera_params.items():
        R, _ = cv2.Rodrigues(np.array([x[0] for x in params['rvec']]))
        t = np.array([x[0] for x in params['tvec']]).reshape(3, 1)
        C = -np.dot(R.T, t)
        positions.append(C.flatten())
        directions.append(R[:, 2])
        names.append(name)
    
    # 创建图形并绘制相机
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 计算关键点的范围来确定合适的视锥体大小
    points_range = np.max(frame_points_3d, axis=0) - np.min(frame_points_3d, axis=0)
    magnitude = np.mean(points_range) * 0.2  # 减小视锥体大小，改为使用关键点范围的20%
    
    # 在同一个图形上绘制相机和关键点
    for pos, dir, name, color in zip(positions, directions, names, plt.cm.jet(np.linspace(0, 1, len(positions)))):
        # 转换相机位置和方向以匹配新的坐标系
        pos_transformed = np.array([pos[0], pos[2], pos[1]])  # 转换相机位置
        dir_transformed = np.array([dir[0], dir[2], dir[1]])  # 转换相机方向
        
        # 绘制相机位置
        ax.scatter(pos_transformed[0], pos_transformed[1], pos_transformed[2], color=color, marker='o')
        ax.text(pos_transformed[0], pos_transformed[1], pos_transformed[2], name, color='black')
        
        # 计算视锥体顶点
        front_point = pos_transformed + dir_transformed * magnitude
        
        # 在新坐标系中计算视锥体的基向量
        up = np.array([0, 0, 1])  # 新的up方向是z轴
        right = np.cross(dir_transformed, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([1, 0, 0])
            right = np.cross(dir_transformed, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, dir_transformed)
        up = up / np.linalg.norm(up)
        
        # 计算视锥体的四个角点
        corners = [
            front_point + (right + up) * magnitude * 0.5,
            front_point + (right - up) * magnitude * 0.5,
            front_point + (-right - up) * magnitude * 0.5,
            front_point + (-right + up) * magnitude * 0.5
        ]
        
        # 绘制锥体
        ax.quiver(pos_transformed[0], pos_transformed[1], pos_transformed[2], 
                 dir_transformed[0], dir_transformed[1], dir_transformed[2], 
                 length=magnitude, color=color, arrow_length_ratio=0.1)
        
        # 绘制视锥体线条
        for i in range(4):
            corner = corners[i]
            ax.plot([pos_transformed[0], corner[0]], 
                   [pos_transformed[1], corner[1]], 
                   [pos_transformed[2], corner[2]], color=color)
            next_corner = corners[(i + 1) % 4]
            ax.plot([corner[0], next_corner[0]], 
                   [corner[1], next_corner[1]], 
                   [corner[2], next_corner[2]], color=color)
    
    # 更新骨架连接定义
    connections = [
        # Head connections
        (0, 1),   # nose - left_eye
        (0, 2),   # nose - right_eye
        (1, 3),   # left_eye - left_ear
        (2, 4),   # right_eye - right_ear

        # Arm connections
        (5, 7),   # left_shoulder - left_elbow
        (7, 9),   # left_elbow - left_wrist
        (6, 8),   # right_shoulder - right_elbow
        (8, 10),  # right_elbow - right_wrist

        # Torso connections
        (3, 5),   # left_ear - left_shoulder
        (4, 6),   # right_ear - right_shoulder
        (5, 6),   # left_shoulder - right_shoulder
        (5, 11),  # left_shoulder - left_hip
        (6, 12),  # right_shoulder - right_hip
        (11, 12), # left_hip - right_hip

        # Leg connections
        (11, 13), # left_hip - left_knee
        (13, 15), # left_knee - left_ankle
        (12, 14), # right_hip - right_knee
        (14, 16), # right_knee - right_ankle
    ]
    
    # 定义不同部位的颜色
    connection_colors = {
        'head': 'red',      # 头部连接
        'arms': 'blue',     # 手臂连接
        'torso': 'green',   # 躯干连接
        'legs': 'purple'    # 腿部连接
    }
    
    # 为不��部位的连接指定颜色
    connection_groups = {
        'head': [(0, 1), (0, 2), (1, 3), (2, 4)],
        'arms': [(5, 7), (7, 9), (6, 8), (8, 10)],
        'torso': [(3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)],
        'legs': [(11, 13), (13, 15), (12, 14), (14, 16)]
    }
    
    # 绘制3D关键点和骨架
    for i, (x, y, z) in enumerate(frame_points_3d):
        ax.scatter(x, y, z, c='black', marker='o', s=100)  # 增大点的大小
        ax.text(x, y, z, str(i), fontsize=10, fontweight='bold')  # 增大标签字体
    
    # 绘制骨架连接，增加线条粗细
    for part, connections_in_group in connection_groups.items():
        color = connection_colors[part]
        for start, end in connections_in_group:
            ax.plot([frame_points_3d[start, 0], frame_points_3d[end, 0]],
                   [frame_points_3d[start, 1], frame_points_3d[end, 1]],
                   [frame_points_3d[start, 2], frame_points_3d[end, 2]],
                   c=color, linewidth=3, alpha=0.8)  # 增加线条粗细
    
    # 调整视角和坐标轴范围
    ax.view_init(elev=20, azim=45)  # 调整视角
    
    # 计算或使用固定的坐标轴范围
    if viz_config.axis_limits is None:
        # 第一帧：计算合适的显示范围
        points_center = np.mean(frame_points_3d, axis=0)
        points_range = np.max(frame_points_3d, axis=0) - np.min(frame_points_3d, axis=0)
        max_range = np.max(points_range) * 0.6  # 减小显示范围，使人体姿态更大
        
        viz_config.axis_limits = {
            'x': [points_center[0] - max_range, points_center[0] + max_range],
            'y': [points_center[1] - max_range, points_center[1] + max_range],
            'z': [points_center[2] - max_range, points_center[2] + max_range]
        }
    
    # 使用固定的坐标轴范围
    ax.set_xlim(viz_config.axis_limits['x'])
    ax.set_ylim(viz_config.axis_limits['y'])
    ax.set_zlim(viz_config.axis_limits['z'])
    
    # 设置网格
    ax.grid(True)
    
    # 可选：设置背景颜色使点更容易看见
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    
    plt.title(f'Frame {frame_idx}')
    
    # 保存图像
    output_dir = "output/3d_visualization"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'))
    plt.close()
