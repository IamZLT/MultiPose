import numpy as np

class SkeletonConstraint:
    def __init__(self):
        # 定义骨架连接关系
        self.connections = [
            # 躯干
            (5, 6),   # left_shoulder - right_shoulder
            (5, 11),  # left_shoulder - left_hip
            (6, 12),  # right_shoulder - right_hip
            (11, 12), # left_hip - right_hip
            
            # 左臂
            (5, 7),   # left_shoulder - left_elbow
            (7, 9),   # left_elbow - left_wrist
            
            # 右臂
            (6, 8),   # right_shoulder - right_elbow
            (8, 10),  # right_elbow - right_wrist
            
            # 左腿
            (11, 13), # left_hip - left_knee
            (13, 15), # left_knee - left_ankle
            
            # 右腿
            (12, 14), # right_hip - right_knee
            (14, 16), # right_knee - right_ankle
            
            # 头部
            (0, 1),   # nose - left_eye
            (0, 2),   # nose - right_eye
            (1, 3),   # left_eye - left_ear
            (2, 4),   # right_eye - right_ear
        ]
        
        # 定义骨骼层级关系（父节点到子节点的方向）
        self.hierarchy = {
            # 躯干作为根部
            5: [7, 11],     # left_shoulder -> [left_elbow, left_hip]
            6: [8, 12],     # right_shoulder -> [right_elbow, right_hip]
            7: [9],         # left_elbow -> left_wrist
            8: [10],        # right_elbow -> right_wrist
            11: [13],       # left_hip -> left_knee
            13: [15],       # left_knee -> left_ankle
            12: [14],       # right_hip -> right_knee
            14: [16],       # right_knee -> right_ankle
            0: [1, 2],      # nose -> [left_eye, right_eye]
            1: [3],         # left_eye -> left_ear
            2: [4],         # right_eye -> right_ear
        }
        
        # 定义标准骨骼长度（可以根据实际数据统计得到）
        self.standard_lengths = {
            # 躯干
            (5, 6): 0.4,   # 肩宽
            (5, 11): 0.5,  # 躯干高度
            (6, 12): 0.5,  # 躯干高度
            (11, 12): 0.3, # 髋宽
            
            # 手臂
            (5, 7): 0.3,   # 上臂
            (7, 9): 0.25,  # 前臂
            (6, 8): 0.3,   # 上臂
            (8, 10): 0.25, # 前臂
            
            # 腿部
            (11, 13): 0.6, # 大腿
            (13, 15): 0.6, # 小腿
            (12, 14): 0.6, # 大腿
            (14, 16): 0.6, # 小腿
            
            # 头部
            (0, 1): 0.08,  # 鼻-眼距离
            (0, 2): 0.08,  # 鼻-眼距离
            (1, 3): 0.1,   # 眼-耳距离
            (2, 4): 0.1,   # 眼-耳距离
        }
        
        # 设置长度约束的容忍度
        self.length_tolerance = 0.1  # 允许10%的长度变化
        
        # 滑动窗口大小
        self.window_size = 30
        # 存储历史长度数据
        self.length_history = {conn: [] for conn in self.connections}
        # 长度约束范围
        self.min_ratio = 0.2  # 最小允许长度为标准长度的20%
        self.max_ratio = 1.2  # 最大允许长度为标准长度的120%
    
    def adjust_bone_length(self, points_3d, parent_idx, child_idx, target_length):
        """调整单个骨骼的长度，考虑最小和最大长度约束"""
        # 获取父节点和子节点的位置
        parent_pos = points_3d[parent_idx]
        child_pos = points_3d[child_idx]
        
        # 计算当前方向向量
        direction = child_pos - parent_pos
        current_length = np.linalg.norm(direction)
        
        if current_length < 1e-6:  # 避免除零错误
            return points_3d
            
        # 计算目标长度（考虑约束）
        min_length = target_length * self.min_ratio
        max_length = target_length * self.max_ratio
        
        if current_length < min_length:
            target_length = min_length
        elif current_length > max_length:
            target_length = max_length
        else:
            return points_3d  # 如果在允许范围内，不进行调整
            
        # 归一化方向向量
        direction = direction / current_length
        
        # 计算新的子节点位置
        new_child_pos = parent_pos + direction * target_length
        
        # 更新点位置
        new_points = points_3d.copy()
        new_points[child_idx] = new_child_pos
        
        return new_points

    def adjust_chain(self, points_3d, start_idx, chain):
        """递归调整骨骼链"""
        adjusted_points = points_3d.copy()
        
        # 处理当前节点的所有子节点
        for child_idx in chain:
            # 获取目标长度
            target_length = self.standard_lengths.get((start_idx, child_idx))
            if target_length is None:
                target_length = self.standard_lengths.get((child_idx, start_idx))
            
            if target_length is not None:
                # 调整当前骨骼长度
                adjusted_points = self.adjust_bone_length(
                    adjusted_points, start_idx, child_idx, target_length)
                
                # 如果子节点还有子节点，继续递归调整
                if child_idx in self.hierarchy:
                    adjusted_points = self.adjust_chain(
                        adjusted_points, child_idx, self.hierarchy[child_idx])
        
        return adjusted_points

    def update_length_constraints(self, points_3d):
        """使用滑动窗口更新骨骼长度约束"""
        # 计算当前帧的所有骨骼长度
        current_lengths = {}
        for i, j in self.connections:
            length = np.linalg.norm(points_3d[i] - points_3d[j])
            if length > 0.01:  # 排除过小的值
                current_lengths[(i, j)] = length
                
                # 更新历史数据
                self.length_history[(i, j)].append(length)
                # 保持窗口大小
                if len(self.length_history[(i, j)]) > self.window_size:
                    self.length_history[(i, j)].pop(0)
        
        # 更新标准长度
        for (i, j) in self.connections:
            if self.length_history[(i, j)]:
                # 使用窗口内的中位数作为标准长度
                median_length = np.median(self.length_history[(i, j)])
                
                # 如果原始标准长度存在，使用它作为参考来限制校准值的范围
                if (i, j) in self.standard_lengths:
                    original_length = self.standard_lengths[(i, j)]
                    # 限制校准值在原始值的±50%范围内
                    median_length = np.clip(
                        median_length,
                        original_length * 0.5,
                        original_length * 1.5
                    )
                self.standard_lengths[(i, j)] = median_length

    def optimize_points(self, points_3d, confidences=None):
        """优化关键点位置"""
        # 首先更新长度约束
        self.update_length_constraints(points_3d)
        
        # 从躯干开始调整整个骨架
        adjusted_points = points_3d.copy()
        
        # 从躯干核心点开始调整
        root_joints = [5, 6]  # left_shoulder, right_shoulder
        for root in root_joints:
            if root in self.hierarchy:
                adjusted_points = self.adjust_chain(
                    adjusted_points, root, self.hierarchy[root])
        
        return adjusted_points

    def check_length_violations(self, points_3d):
        """检查骨骼长度违反情况"""
        violations = []
        for i, j in self.connections:
            current_length = np.linalg.norm(points_3d[i] - points_3d[j])
            target_length = self.standard_lengths.get((i, j)) or self.standard_lengths.get((j, i))
            
            if target_length:
                min_length = target_length * self.min_ratio
                max_length = target_length * self.max_ratio
                
                if current_length < min_length or current_length > max_length:
                    violations.append({
                        'connection': (i, j),
                        'current_length': current_length,
                        'target_length': target_length,
                        'min_length': min_length,
                        'max_length': max_length,
                        'error_ratio': abs(current_length - target_length) / target_length
                    })
        return violations