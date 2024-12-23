import numpy as np

class KeypointKalmanFilter:
    def __init__(self, num_keypoints=17):
        """初始化关键点的卡尔曼滤波器
        
        状态向量 x = [x, y, z, vx, vy, vz]
        每个关键点有一个独立的滤波器
        """
        self.num_keypoints = num_keypoints
        self.filters = []
        
        for _ in range(num_keypoints):
            # 为每个关键点创建一个滤波器
            self.filters.append({
                'x': np.zeros((6, 1)),  # 状态向量 [x, y, z, vx, vy, vz]
                'P': np.eye(6) * 1000,  # 初始协方差矩阵
                'initialized': False
            })
        
        # 状态转移矩阵 F
        dt = 1.0  # 时间步长
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 测量矩阵 H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(6) * 0.1
        
        # 测量噪声协方差矩阵 R
        self.R = np.eye(3) * 1.0

    def update(self, points_3d, confidences=None):
        """更新所有关键点的状态
        
        Args:
            points_3d: shape (num_keypoints, 3) 的numpy数组
            confidences: shape (num_keypoints,) 的置信度数组
        
        Returns:
            filtered_points: shape (num_keypoints, 3) 的滤波后的点
        """
        filtered_points = np.zeros_like(points_3d)
        
        for i in range(self.num_keypoints):
            point = points_3d[i]
            confidence = confidences[i] if confidences is not None else 1.0
            
            # 根据置信度调整测量噪声
            R = self.R / max(confidence, 0.1)
            
            if not self.filters[i]['initialized']:
                # 初始化状态
                self.filters[i]['x'][:3, 0] = point
                self.filters[i]['initialized'] = True
                filtered_points[i] = point
                continue
            
            # 预测步骤
            x_pred = self.F @ self.filters[i]['x']
            P_pred = self.F @ self.filters[i]['P'] @ self.F.T + self.Q
            
            # 更新步骤
            z = point.reshape(3, 1)
            y = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            
            # 更新状态
            self.filters[i]['x'] = x_pred + K @ y
            self.filters[i]['P'] = (np.eye(6) - K @ self.H) @ P_pred
            
            # 保存滤波后的位置
            filtered_points[i] = self.filters[i]['x'][:3, 0]
        
        return filtered_points 