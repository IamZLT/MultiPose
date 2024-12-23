import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rodrigues_vec_to_rotation_mat(rvec):
    """将Rodrigues旋转向量转换为旋转矩阵"""
    rvec = np.array(rvec, dtype=np.float32).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R

def get_camera_center_world(R, t):
    """
    计算相机在世界坐标系中的位置，并进行坐标系变换
    相机中心在相机坐标系中是原点[0,0,0]，需要转换到世界坐标系
    C = -R^T * t
    """
    R = np.array(R)
    t = np.array(t)
    C = -np.dot(R.T, t)
    # 将y轴变为z轴的变换矩阵
    transform_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],  # y -> z
        [0, 1, 0]   # z -> y
    ])
    C = np.dot(transform_matrix, C)
    return C.flatten()

def get_camera_direction(R):
    """从旋转矩阵获取相机朝向（光轴方向），并进行坐标系变换"""
    direction = R[:, 2]
    # 将y轴变为z轴的变换矩阵
    transform_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],  # y -> z
        [0, 1, 0]   # z -> y
    ])
    direction = np.dot(transform_matrix, direction)
    return direction

def visualize_cameras(positions, directions, names, magnitude=1.0):
    """
    在3D空间中可视化相机位置和朝向
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 定义颜色列表
    colors = plt.cm.jet(np.linspace(0, 1, len(positions)))
    
    # 绘制每个相机的位置和方向
    for pos, dir, name, color in zip(positions, directions, names, colors):
        pos = np.array(pos)
        dir = np.array(dir)
        
        # 绘制相机位置
        ax.scatter(pos[0], pos[1], pos[2], color=color, marker='o')
        ax.text(pos[0], pos[1], pos[2], name, color='black')
        
        # 计算视锥体顶点
        front_point = pos + dir * magnitude
        
        # 修改基向量的定义，使用新的坐标系
        up = np.array([0, 1, 0])  # 在新坐标系中，up方向是y轴
        right = np.cross(dir, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([1, 0, 0])
            right = np.cross(dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, dir)
        up = up / np.linalg.norm(up)
        
        # 计算视锥体的四个角点
        corners = [
            front_point + (right + up) * magnitude * 0.5,
            front_point + (right - up) * magnitude * 0.5,
            front_point + (-right - up) * magnitude * 0.5,
            front_point + (-right + up) * magnitude * 0.5
        ]
        
        # 绘制从相机位置到前方点的线，并加上箭头
        ax.quiver(pos[0], pos[1], pos[2], 
                 dir[0], dir[1], dir[2], 
                 length=magnitude, 
                 color=color, 
                 arrow_length_ratio=0.1,
                 normalize=True)  # 添加normalize参数
        
        # 绘制视锥体的边
        for i in range(4):
            # 连接相机位置和角点
            corner = corners[i]
            ax.plot([pos[0], corner[0]], [pos[1], corner[1]], [pos[2], corner[2]], color=color)
            
            # 连接相邻角点
            next_corner = corners[(i + 1) % 4]
            ax.plot([corner[0], next_corner[0]], 
                   [corner[1], next_corner[1]], 
                   [corner[2], next_corner[2]], color=color)
    
    # 设置合适的视角
    ax.view_init(elev=30, azim=45)
    
    # 调整坐标轴比例使其相等
    max_range = np.array([
        ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
        ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
        ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim3d()[1] + ax.get_xlim3d()[0]) / 2.0
    mid_y = (ax.get_ylim3d()[1] + ax.get_ylim3d()[0]) / 2.0
    mid_z = (ax.get_zlim3d()[1] + ax.get_zlim3d()[0]) / 2.0
    
    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
    
    plt.title('Camera Positions and Directions')
    
    # 保存图像
    output_dir = "output/camera_visualization"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'camera_poses.png'), dpi=300, bbox_inches='tight')
    plt.close()

def read_camera_params(parquet_file):
    df = pd.read_parquet(parquet_file)
    
    # 数据验证
    required_columns = ['name', 'rvec', 'tvec', 'camera_matrix', 'distortion_coefficients']
    if not all(col in df.columns for col in required_columns):
        print("Error: Missing required columns in parquet file")
        return None
    
    positions = []
    directions = []
    names = []
    
    # 处理每个相机的参数
    for index, row in df.iterrows():
        print(f"\n{'='*50}")
        print(f"Camera {row['name']}:")
        
        # 旋转信息
        rvec = np.array([x[0] for x in row['rvec']])
        R = rodrigues_vec_to_rotation_mat(rvec)
        
        # 平移向量
        t = np.array([x[0] for x in row['tvec']]).reshape(3,1)
        
        # 计算相机中心在世界坐标系中的位置
        C = get_camera_center_world(R, t)
        print(f"\nCamera Center in World Coordinates (X,Y,Z):")
        print(C)
        positions.append(C)
        
        # 计算相机朝向
        direction = get_camera_direction(R)
        print(f"\nCamera Direction (X,Y,Z):")
        print(direction)
        directions.append(direction)
        
        names.append(row['name'])
        
        # 打印内参矩阵
        K = np.array([x for x in row['camera_matrix']]).reshape(3,3)
        print(f"\nIntrinsic Matrix:")
        print(K)
        
        # 打印畸变系数
        dist = np.array(row['distortion_coefficients'][0])
        print(f"\nDistortion Coefficients (k1,k2,p1,p2,k3):")
        print(dist)
    
    # 可视化相机位置和朝向
    visualize_cameras(positions, directions, names)
    
    return df


def get_projection_matrix(camera_params):
    """从相机参数计算投影矩阵"""
    # 获取相机内参矩阵
    camera_matrix = camera_params['camera_matrix']
    K = np.array([
        camera_matrix[0],  # 第一行
        camera_matrix[1],  # 第二行
        camera_matrix[2]   # 第三行
    ])
    
    # 获取旋转向量和平移向量
    rvec = np.array([x[0] for x in camera_params['rvec']])
    tvec = np.array([x[0] for x in camera_params['tvec']])
    
    # 计算旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    # 构建投影矩阵 P = K[R|t]
    Rt = np.hstack((R, tvec.reshape(3, 1)))
    P = np.dot(K, Rt)
    
    return P


if __name__ == "__main__":
    parquet_file = "sample/data-1219/params.parquet"
    if not os.path.exists(parquet_file):
        print(f"Error: File {parquet_file} does not exist!")
        exit(1)
    
    camera_params = read_camera_params(parquet_file)