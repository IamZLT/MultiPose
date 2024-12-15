import numpy as np

def get_velocity_at_this_timestamp_for_this_id_for_cur_timestamp(poses_3d_all_timestamps, timestamp_latest_pose, points_3d_latest_pose, id_latest_pose, delta_time_threshold = 0.1):
    """Calculate velocity for given pose ID"""
    velocity_t = np.zeros((len(points_3d_latest_pose)))
    timestamp_tilde_frame = 0.0
    
    for index in range(len(poses_3d_all_timestamps)-1,0,-1):
        this_timestamp = list(poses_3d_all_timestamps.keys())[index]
        if (timestamp_latest_pose - this_timestamp) > delta_time_threshold:
            break
        if this_timestamp >= timestamp_latest_pose or all(value is None for value in poses_3d_all_timestamps[this_timestamp]):
            continue
        for id_index in range(len(poses_3d_all_timestamps[this_timestamp])):
            if poses_3d_all_timestamps[this_timestamp][id_index]['id'] == id_latest_pose:
                points_3d_tilde_timestamp = np.array(poses_3d_all_timestamps[this_timestamp][id_index]['points_3d'])
                timestamp_tilde_frame = this_timestamp
                break
    
    if timestamp_tilde_frame > 0 and (timestamp_latest_pose > timestamp_tilde_frame):
        assert len(points_3d_latest_pose) == len(points_3d_tilde_timestamp)
        
        for k in range(len(points_3d_latest_pose)):
            p_x1, p_y1, p_z1 = points_3d_latest_pose[k]
            p_x2, p_y2, p_z2 = points_3d_tilde_timestamp[k]
            displacement_t = (p_x1 - p_x2) + (p_y1 - p_y2) + (p_z1 - p_z2)
            velocity_t[k] = displacement_t / (float(timestamp_latest_pose) - float(timestamp_tilde_frame))
         
    return velocity_t.tolist()

def get_latest_3D_poses_available_for_cur_timestamp(poses_3d_all_timestamps, timestamp_cur_frame, delta_time_threshold = 0.1):
    """Get latest 3D poses within time window"""
    poses_3D_latest = []
    id_list = []
    
    for index in range(len(poses_3d_all_timestamps)-1,0,-1):
        this_timestamp = list(poses_3d_all_timestamps.keys())[index]
        if (timestamp_cur_frame - this_timestamp) > delta_time_threshold:
            break
        if this_timestamp >= timestamp_cur_frame or all(value is None for value in poses_3d_all_timestamps[this_timestamp]):
            continue
        if all(value is not None for value in poses_3d_all_timestamps[this_timestamp]):
            for id_index in range(len(poses_3d_all_timestamps[this_timestamp])):
                if poses_3d_all_timestamps[this_timestamp][id_index]['id'] not in id_list:
                    poses_3D_latest.append({
                        'id': poses_3d_all_timestamps[this_timestamp][id_index]['id'],
                        'points_3d': poses_3d_all_timestamps[this_timestamp][id_index]['points_3d'],
                        'timestamp': this_timestamp,
                        'velocity': get_velocity_at_this_timestamp_for_this_id_for_cur_timestamp(
                            poses_3d_all_timestamps,
                            this_timestamp,
                            poses_3d_all_timestamps[this_timestamp][id_index]['points_3d'],
                            poses_3d_all_timestamps[this_timestamp][id_index]['id'])
                    })
                    id_list.append(poses_3d_all_timestamps[this_timestamp][id_index]['id'])
    
    if len(poses_3D_latest)>0:
        poses_3D_latest = sorted(poses_3D_latest, key=lambda i: int(i['id']), reverse=False)
    return poses_3D_latest 