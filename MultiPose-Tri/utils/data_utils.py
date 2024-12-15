import numpy as np

def calculate_perpendicular_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line"""
    distance = np.linalg.norm(np.cross(line_end-line_start, line_start-point))/np.linalg.norm(line_end-line_start)
    return distance

def separate_lists_for_incremental_triangulation(data):
    """Separate detection data into lists by camera"""
    result = {}
    for item in data:
        for key, value in item.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    return result

def extract_key_value_pairs_from_poses_2d_list(data, id, timestamp_cur_frame, delta_time_threshold = 0.1):
    """Extract pose data for given ID within time window"""
    camera_id_covered_list = []
    result = []
    
    for index in range(len(data)-1,0,-1):
        this_timestamp = data[index]['timestamp']
        this_camera = data[index]['camera']
        
        if (timestamp_cur_frame - this_timestamp) > delta_time_threshold:
            break
            
        if this_camera not in camera_id_covered_list:
            for pose_index in range(len(data[index]['poses'])):
                if data[index]['poses'][pose_index]['id'] == id:
                    result.append({
                        'camera': this_camera,
                        'timestamp': this_timestamp,
                        'poses': data[index]['poses'][pose_index],
                        'image_wh': data[index]['image_wh']
                    })
                    camera_id_covered_list.append(this_camera)
                    break
    
    return result 