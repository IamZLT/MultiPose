import torch
import numpy as np






def homogeneous_to_euclidean(points):
    """将齐次坐标转换为欧几里得坐标"""
    return points[..., :-1] / points[..., -1:]

def triangulate_points_with_confidence(points1, points2, P1, P2, scores1=None, scores2=None):
    """使用置信度的改进版三角测量函数"""
    # 转换为PyTorch张量
    points1 = torch.from_numpy(points1.astype(np.float32))
    points2 = torch.from_numpy(points2.astype(np.float32))
    P1 = torch.from_numpy(P1.astype(np.float32))
    P2 = torch.from_numpy(P2.astype(np.float32))
    
    if scores1 is not None and scores2 is not None:
        scores1 = torch.from_numpy(scores1.astype(np.float32))
        scores2 = torch.from_numpy(scores2.astype(np.float32))
        # 修改这里：不再stack置信度
        confidences = [scores1, scores2]
    else:
        confidences = None
    
    # 准备投影矩阵
    proj_matrices = torch.stack([P1, P2], dim=0)
    
    # 存储所有关键点的3D坐标
    points_3d = []
    
    # 对每个关键点进行三角测量
    for i in range(len(points1)):
        points = torch.stack([points1[i], points2[i]], dim=0)
        if confidences is not None:
            # 只取当前关键点的置信度
            conf = torch.tensor([confidences[0][i], confidences[1][i]], dtype=torch.float32)
        else:
            conf = None
        
        # 对单个点进行三角测量
        point_3d = triangulate_point_from_multiple_views_linear_torch(
            proj_matrices, 
            points,
            conf
        )
        points_3d.append(point_3d)
    
    # 转换回numpy数组
    points_3d = torch.stack(points_3d).numpy()
    
    # 进行坐标轴变换：(x,y,z) -> (x,z,y)
    points_3d_transformed = points_3d.copy()
    points_3d_transformed[:, 1] = points_3d[:, 2]
    points_3d_transformed[:, 2] = points_3d[:, 1]
    
    return points_3d_transformed

def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """多视图三角测量的PyTorch实现
    
    Args:
        proj_matricies: 形状为(N, 3, 4)的投影矩阵序列
        points: 形状为(N, 2)的点坐标序列
        confidences: 形状为(N,)的置信度序列，范围[0.0, 1.0]
    
    Returns:
        point_3d: 形状为(3,)的三角测量得到的3D点
    """
    assert len(proj_matricies) == len(points)
    
    n_views = len(proj_matricies)
    
    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32)
    
    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    
    # 修改置信度的应用方式
    confidences = confidences.view(-1, 1, 1)
    A = A * confidences
    
    u, s, vh = torch.svd(A.view(-1, 4))
    
    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]
    
    return point_3d
