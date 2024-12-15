import numpy as np

def compute_affinity_epipolar_constraint_with_pairs(detections_pairs, alpha_2D, calibration):
    """
    Compute affinity between two detections using epipolar constraint
    
    Args:
        detections_pairs: Tuple of two detections to compare
        alpha_2D: Threshold for 2D affinity
        calibration: Calibration object containing camera parameters
    """
    Au_this_pair = 0
    
    D_L = np.array(detections_pairs[0]['points_2d'])
    D_R = np.array(detections_pairs[1]['points_2d'])
    cam_L_id = detections_pairs[0]['camera_id']
    cam_R_id = detections_pairs[1]['camera_id']
    
    assert len(D_L)==len(D_R)
    assert cam_L_id != cam_R_id
    F_matrix = calibration.get_fundamental_matrix([cam_L_id,cam_R_id])
    
    Au_this_pair = (1 - ((calibration.distance_between_epipolar_lines(D_L, D_R, F_matrix))/ (2*alpha_2D)))
    
    return Au_this_pair

def get_affinity_matrix_epipolar_constraint(Du, alpha_2D, calibration):
    """
    Compute affinity matrix for all detection pairs using epipolar constraint
    
    Args:
        Du: List of unmatched detections
        alpha_2D: Threshold for 2D affinity
        calibration: Calibration object containing camera parameters
    """
    Du_cam_wise_split = {}
    for entry in Du:
        camera_id = entry['camera_id']
        if camera_id not in Du_cam_wise_split:
            Du_cam_wise_split[camera_id] = []
        Du_cam_wise_split[camera_id].append(entry)
    
    num_entries = sum(len(entries) for entries in Du_cam_wise_split.values())
    Au = np.zeros((num_entries, num_entries), dtype=np.float32)
    
    camera_id_to_index = {camera_id: i for i, camera_id in enumerate(Du_cam_wise_split.keys())}
    
    for camera_id, entries in Du_cam_wise_split.items():
        for i in range(len(entries)):
            for other_camera_id, other_entries in Du_cam_wise_split.items():
                if other_camera_id != camera_id:
                    for j in range(len(other_entries)):
                        pair_ij = (entries[i], other_entries[j])
                        pair_ji = (other_entries[j], entries[i])
                        index_i = camera_id_to_index[camera_id] * len(entries) + i
                        index_j = camera_id_to_index[other_camera_id] * len(other_entries) + j                        
                        
                        Au[index_i, index_j] = compute_affinity_epipolar_constraint_with_pairs(
                            pair_ij, alpha_2D, calibration)
                        Au[index_j, index_i] = compute_affinity_epipolar_constraint_with_pairs(
                            pair_ji, alpha_2D, calibration)
                        
    return Au