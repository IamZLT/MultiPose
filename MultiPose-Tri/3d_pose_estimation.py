# %%

import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 然后使用绝对导入
import argparse
import cv2
import numpy as np
import pandas as pd
from numpy.linalg import svd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from dataset.crossview import data_utils
from dataset.crossview.visualization.visualizer import Visualizer
from dataset.crossview.calib.calibration import Calibration
from matplotlib.patches import Circle

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

from scipy.optimize import linear_sum_assignment
from collections import defaultdict, OrderedDict

from scipy.sparse import coo_matrix
from core.bip_solver import GLPKSolver

import logging
#from tqdm import tqdm
from tqdm.notebook import tqdm

from visualization.pose_visualizer import (
    show_cam_image_with_pose_keypoints,
    show_cam_location_on_3d_plot
)
from core.pose_matcher import (
    compute_affinity_epipolar_constraint_with_pairs,
    get_affinity_matrix_epipolar_constraint
)
from core.pose_estimator import (
    get_velocity_at_this_timestamp_for_this_id_for_cur_timestamp,
    get_latest_3D_poses_available_for_cur_timestamp
)
from utils.data_utils import (
    calculate_perpendicular_distance,
    separate_lists_for_incremental_triangulation,
    extract_key_value_pairs_from_poses_2d_list
)



# %%
# path to the dataset. Change accordingly
FOLDER = "../Campus_Seq1/"
frame_root = os.path.join(FOLDER,"frames")
calibration_file = os.path.join(FOLDER,"calibration.json") 
pose_file = os.path.join(FOLDER,"annotation_2d.json")

# frame iterator object
frame_loader = data_utils.FrameLoader(frame_root, None)
# pose iterator object
pose_loader = data_utils.Pose2DLoader(pose_file)

# load calibration
calibration = Calibration.from_json(calibration_file)


# change only after looking at the code below
delta_time_threshold = 0.1  

# config used in the paper for the campus dataset 

# 2D correspondence config
w_2D = 0.4  # Weight of 2D correspondence
alpha_2D = 25 # Threshold of 2D velocity
lambda_a = 5  # Penalty rate of time interval
lambda_t = 10

# 3D correspondence confif
w_3D = 0.6  # Weight of 3D correspondence
alpha_3D = 0.1  # Threshold of distance
#-----------------------------------------------------
## Configure logger
logging.basicConfig(filename='CVT_for_3d_pose_reconstruction.log',encoding='utf-8',level=logging.INFO, filemode = 'w') 

wait_time = 1

no_of_cameras = len(calibration.camera_ids)
world_ltrb = calibration.compute_world_ltrb()

num_body_joints_detected_by_2d_pose_detector = 14
# camera location
camera_id_list = list(calibration.camera_ids)
camera_3d_location = [] 
camera_look_at = []
for cam_index in range(no_of_cameras):
    cam_loc = calibration.cameras[camera_id_list[cam_index]].location
    camera_3d_location.append(cam_loc)
    camera_look_at.append(calibration.cameras[camera_id_list[cam_index]].look_at)
    print(f'Camera id {camera_id_list[cam_index]} | location: {cam_loc}')

# %%
## Vizualization functions

# list of dictionary per frame/iteration with   [{'id': assigned by pose detector, 'points_2d': list of target joints} , {},...,{N}]
poses_2d_all_frames = []

# list of per frame/iteration dictionary with   [{'id': calculated, 'points_3d': list of target joints} , {},...,{M}, 'camera_ID': list of camera ID used for this pose]
poses_3d_all_timestamps = defaultdict(list) 

# list of dictionary per frame/iteration with   [{'camera_id': , 'points_2d': list of target joints} , {},...,{<=N}]
unmatched_detections_all_frames = defaultdict(list) 


new_id = -1 # So that ID number starts from 0
iterations = 0  #  
new_id_last_update_timestamp = 0  # latest timestamp at which we detected a new person.


#### Loop through all the camera frames one by one

for frame_index in tqdm(range(len(frame_loader))):
    
    #### (1) Intialize variable for each iterations
    indices_T = [] 
    indices_D = []
    logging.info('----'*100)
    logging.info(iterations)
    
    #### (2) Get and preprocess data to use in algo. 
    frame_cam = frame_loader.__getitem__(frame_index)
    camera_id_cur_frames = frame_cam['camera_name']
    timestamp_cur_frame = frame_cam['timestamp']
    logging.info(f'Current timestamp: {timestamp_cur_frame}')
    
    ## get 2D poses of all detections in the curent frames
    if pose_file:
        
        points_2d_cur_frames = []
        points_2d_scores_cur_frames = []
        pose_json_cur_frame = pose_loader.get_data(frame_cam["frame_name"].replace("\\", "/"))
        poses_cur_frames = pose_json_cur_frame['poses']   # poses list with multiple ids + points_2d + scores
        
        if not poses_cur_frames:
            
            iterations+=1
            poses_3d_all_timestamps[timestamp_cur_frame].append(None)
            continue 
        
        for poses_index in range(len(poses_cur_frames)):
            #print(poses_cur_frames[poses_index]['id'])
            logging.info(f'Original ID : {poses_cur_frames[poses_index]["id"]}')
            # deleting ID as they are already assigned in the current dataset
            poses_cur_frames[poses_index]['id'] = '-1'
            points_2d_cur_frames.append(poses_cur_frames[poses_index]['points_2d'])
            points_2d_scores_cur_frames.append(poses_cur_frames[poses_index]['scores'])
            
        image_wh_cur_frames = pose_json_cur_frame['image_wh']
        location_camera_center_cur_frames = calibration.cameras[camera_id_cur_frames].location
        #print(f'poses cur frames: {poses_cur_frames}')
        logging.info(f'poses cur frames: {poses_cur_frames}')
    # Use pose detector on current frame YOLO!
    # Currently not implemented     
    else:
        continue
    
    poses_2d_all_frames.append(pose_json_cur_frame)
    
    
    ## get all available 3D poses from the last timestamp
    poses_3D_latest = get_latest_3D_poses_available_for_cur_timestamp(poses_3d_all_timestamps, timestamp_cur_frame, delta_time_threshold = delta_time_threshold)
    #print(f'poses 3d for this iter: {poses_3D_latest}')
    logging.info(f'poses 3d for this iter: {poses_3D_latest}')
    N_3d_poses_last_timestamp = len(poses_3D_latest)
    M_2d_poses_this_camera_frame = len(points_2d_cur_frames)
    
    Dt_c = np.array(points_2d_cur_frames)  # Shape (M poses on frame , no of body points , 2)
    Dt_c_scores = np.array(points_2d_scores_cur_frames)
        
    A = np.zeros((N_3d_poses_last_timestamp, M_2d_poses_this_camera_frame))  # Cross-view association matrix shape N x M
    
    # Cross-view association
    for i in range(N_3d_poses_last_timestamp): # Iterate through prev N Target poses
        
        # x_t_tilde_tilde_c: projection of prev. detected ith 3d pose on a camera with ID camera_id_cur_frames 
        x_t_tilde_tilde_c = calibration.project(np.array(poses_3D_latest[i]['points_3d']), camera_id_cur_frames)
        delta_t = timestamp_cur_frame - poses_3D_latest[i]['timestamp']     # Time interval
        #print(f' delta_t: {delta_t}')
        logging.info(f' delta_t: {delta_t}')
        
        for j in range(M_2d_poses_this_camera_frame): # Iterate through M poses 
            
            # Each detection (Dj_t_c) in this frame will have k body points for every camera c
            # x_t_c in image coordinates
            # x_t_c_norm scale normalized image coordinates
            x_t_c_norm = Dt_c[j].copy() 
            x_t_c_norm[:,0] = x_t_c_norm[:,0] / image_wh_cur_frames[0]
            x_t_c_norm[:,1] = x_t_c_norm[:,1] / image_wh_cur_frames[1]
            
            K_joints_detected_this_person = len(x_t_c_norm)
            
            # use x_t_c vs x_t_c_norm? Verify...
            back_proj_x_t_c_to_ground = calibration.cameras[camera_id_cur_frames].back_project(x_t_c_norm, z_worlds=np.zeros(K_joints_detected_this_person))
            
            for k in range(K_joints_detected_this_person):  # Iterate through K keypoints
                
                distance_2D = np.linalg.norm(x_t_c_norm[k] - x_t_tilde_tilde_c[k])  # Distance between joints

                A_2D = w_2D * (1 - distance_2D / (alpha_2D*delta_t)) * np.exp(-lambda_a * delta_t)
                
                ## TODO: verify velocity estimation...
                #  3D velocity to be estimated via a linear least-square method
                velocity_t_tilde = poses_3D_latest[i]['velocity'][k]
                
                predicted_X_t = np.array(poses_3D_latest[i]['points_3d'][k]) + velocity_t_tilde * delta_t
                
                # Assuming that cameras will be pointed at the ground with z = 0
                # 3d distance between vector given by camera center and ground point to predicted x_t
                # All of the points lie in the same coordinate system? Verify...
                dl = calculate_perpendicular_distance(point = predicted_X_t , line_start = location_camera_center_cur_frames , line_end = back_proj_x_t_c_to_ground[k])
                
                A_3D = w_3D * (1 - dl / alpha_3D) * np.exp(-lambda_a * delta_t)
                
                A[i,j] += A_2D + A_3D
                
    # Perform Hungarian algorithm for assignment for each camera
    indices_T, indices_D = linear_sum_assignment(A, maximize = True)
    
    #print(f'Indices_T, Indices_D: {indices_T, indices_D}')
    logging.info(f'Indices_T, Indices_D: {indices_T, indices_D}')
    # redundant but checking one to one mapping
    assert len(indices_D) == len(indices_T), "number of detection should be equal to target for each iterations"
    
    # Target update
    for i,j in zip(indices_T, indices_D):
        
        # Update latest detections in the poses_2d_all_frames  
        poses_2d_all_frames[-1]['poses'][j]['id'] = poses_3D_latest[i]['id']

        # Get latest 2D Pose data from all the cameras for the detected ID 
        poses_2d_inc_rec_other_cam = extract_key_value_pairs_from_poses_2d_list(poses_2d_all_frames, 
                                                                                id = poses_3D_latest[i]['id'],
                                                                                timestamp_cur_frame = timestamp_cur_frame,
                                                                                delta_time_threshold = delta_time_threshold )
        
        # move following code in func extract_key_value_pairs_from_poses_2d_list to get *_inc_rec variables directly
        # Get 2D poses of ID 
        dict_with_poses_for_n_cameras_for_latest_timeframe = separate_lists_for_incremental_triangulation(poses_2d_inc_rec_other_cam)
        
        camera_ids_inc_rec = []
    
        image_wh_inc_rec = []
    
        timestamps_inc_rec = []

        points_2d_inc_rec = []
        
        camera_ids_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['camera']
        image_wh_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['image_wh']
        timestamps_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['timestamp']
        
        for dict_index in range(len(dict_with_poses_for_n_cameras_for_latest_timeframe['poses'])):
            
            points_2d_inc_rec.append(dict_with_poses_for_n_cameras_for_latest_timeframe['poses'][dict_index]['points_2d'])
        
        # migration to func ends here 
   
        K_joints_detected_this_person = len(Dt_c[j])
        
        Ti_t = []
        
        for k in range(K_joints_detected_this_person): # iterate through k points
            
            # get all the 2d pose point from all the cameras where this target was detected last
            # i.e. if current frame is from cam 1 then get last detected 2d pose of this target 
            # from all of the cameras. Do triangulation with all cameras with detected ID
            
            _, Ti_k_t = calibration.linear_ls_triangulate_weighted(np.array(points_2d_inc_rec)[:,k,:], 
                                                                    camera_ids_inc_rec, 
                                                                    image_wh_inc_rec, 
                                                                    lambda_t, 
                                                                    timestamps_inc_rec)
            Ti_t.append(Ti_k_t.tolist())
        
        
        # If there is no entry for the current id at the current timestamp
        if i >= len(poses_3d_all_timestamps[timestamp_cur_frame]):
            poses_3d_all_timestamps[timestamp_cur_frame].append({'id': poses_3D_latest[i]['id'],
                                                        'points_3d': Ti_t,
                                                        'camera_ID': [camera_id_cur_frames]})
            
        # If there exist an entry already overwrite as this would be contain updated timestamps
        # from all cameras for points 3D. 
        else:
            poses_3d_all_timestamps[timestamp_cur_frame][i]['points_3d'] = Ti_t
            poses_3d_all_timestamps[timestamp_cur_frame][i]['camera_ID'].append(camera_id_cur_frames)
    
    # Try to have same format as points_2d 
    # Target initialization
    
    for j in range(M_2d_poses_this_camera_frame):
        if j not in indices_D:
            unmatched_detections_all_frames[timestamp_cur_frame].append({'camera_id': camera_id_cur_frames,
                                                            'points_2d': Dt_c[j],
                                                            'scores': Dt_c_scores[j],
                                                            'image_wh': image_wh_cur_frames})
    
    # There is no previous 3D target for two timestamps
    ## TODO:
    # Memory of only two timestamps is not good. Change to memory of entire runtime? 
    # Think about corner as here only affinity is geometric consistency between two views
    # and not apperance

    iterations+=1
    
    # Assuming we get frame data for all the cameras sequentially
    if iterations % no_of_cameras == 0:
        # If the list is not empty
        if unmatched_detections_all_frames[timestamp_cur_frame]:
        # delete entries which ever are already used in matching
    
            unique_cameras_set_this_iter_with_unmatched_det = set(item['camera_id'] for item in unmatched_detections_all_frames[timestamp_cur_frame])
            
            num_cameras_this_iter_with_unmatched_det = len(unique_cameras_set_this_iter_with_unmatched_det)
            
            #print(f'num_cameras_this_iter_with_unmatched_det: {num_cameras_this_iter_with_unmatched_det}')
            logging.info(f'num_cameras_this_iter_with_unmatched_det: {num_cameras_this_iter_with_unmatched_det}')
            # if there is unmatched detection from atleast two different cameras for the current timestamp
            if (num_cameras_this_iter_with_unmatched_det > 1):
                
                Au = get_affinity_matrix_epipolar_constraint(
                    unmatched_detections_all_frames[timestamp_cur_frame],
                    alpha_2D,
                    calibration
                )
                #clusters = graph_partition(Au)  # Perform graph partitioning
                solver = GLPKSolver(min_affinity=0, max_affinity=1)
                clusters, sol_matrix = solver.solve(Au.astype(np.double), rtn_matrix  = True)
                
                
                # Target initialization from clusters
                for Dcluster in clusters:
                    points_2d_this_cluster = []
                    camera_id_this_cluster = []
                    image_wh_this_cluster = []
                    
                    if len(Dcluster) >= 2:
                    
                        #print(f'Inside cluster: {Dcluster} ')
                        logging.info(f'Inside cluster: {Dcluster} ')
                        
                        # TODO: Adhoc Solution. Change in the future
                        # If there a new person detected within delta time threshold then probably
                        # this new person is belongs to the older id
                        if timestamp_cur_frame - new_id_last_update_timestamp > delta_time_threshold:
                            
                            new_id_last_update_timestamp = timestamp_cur_frame
                            new_id +=1
                        
                        for detection_index in Dcluster:
                            points_2d_this_cluster.append(unmatched_detections_all_frames[timestamp_cur_frame][detection_index]['points_2d'])
                            
                            camera_id_this_cluster.append(unmatched_detections_all_frames[timestamp_cur_frame][detection_index]['camera_id'])
                            
                            image_wh_this_cluster.append(unmatched_detections_all_frames[timestamp_cur_frame][detection_index]['image_wh'])
                            
                            # Change ID for all the used points in poses 2D all frames for the current timestamp 
                            # Since points are added in order of the original poses_2d_all_frames thus simply 
                            # overwrite the ID to index of the Dcluster. Verify...
                        
                            for new_index_set_id in range(len(poses_2d_all_frames[-len(Dcluster):][detection_index]['poses'])):
                                if str(poses_2d_all_frames[-len(Dcluster):][detection_index]['poses'][new_index_set_id]['id']) == '-1':
                                    poses_2d_all_frames[-len(Dcluster):][detection_index]['poses'][new_index_set_id]['id'] = new_id
                        
                        # Overwriting the unmatched detection for the current timeframe with the indcies not present in the detection cluster
                        unmatched_detections_all_frames[timestamp_cur_frame] = [unmatched_detections_all_frames[timestamp_cur_frame][i] for i in range(len(unmatched_detections_all_frames[timestamp_cur_frame])) if i not in Dcluster]
                        
                        Tnew_t = calibration.triangulate_complete_pose(points_2d_this_cluster,camera_id_this_cluster,image_wh_this_cluster)
                        
                        # Add the 3D points according to the ID 
                        poses_3d_all_timestamps[timestamp_cur_frame].append({'id': new_id,
                                                                    'points_3d': Tnew_t.tolist(),
                                                                    'camera_ID': camera_id_this_cluster})
    
    
    #print(f'unmatched_detections_all_frames: {unmatched_detections_all_frames}')
    logging.info(f'unmatched_detections_all_frames: {unmatched_detections_all_frames}')
    #print(f'poses 3d calc for this timestamp: {poses_3d_all_timestamps[timestamp_cur_frame]}')
    logging.info(f'poses 3d calc for this timestamp: {poses_3d_all_timestamps[timestamp_cur_frame]}')


# %%
ori_wcx = np.mean(world_ltrb[0::2])
ori_wcy = np.mean(world_ltrb[1::2])
world_ltrb_mean_cen = world_ltrb.copy()
world_ltrb_mean_cen[0::2] -= ori_wcx
world_ltrb_mean_cen[1::2] -= ori_wcy
camera_id_list = list(camera_id_list)

#maximum_person_to_plot = 25
#cmap = plt.get_cmap('viridis')
#slicedCM = cmap(np.linspace(0, 1, maximum_person_to_plot)) 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(world_ltrb_mean_cen[0], world_ltrb_mean_cen[2])
ax.set_ylim3d(world_ltrb_mean_cen[1], world_ltrb_mean_cen[3])
ax.set_zlim3d(0, 300)
timestamp_to_plot = float(41.72)
ax.set_title(f'Timestamp: {timestamp_to_plot}')
show_cam_location_on_3d_plot(camera_id_list, calibration, magnitude=20, ax = ax)
scatter_points_arr = []
person_ID_list = []
for pose_index in range(len(poses_3d_all_timestamps[timestamp_to_plot])):
    this_pose = np.array(poses_3d_all_timestamps[timestamp_to_plot][pose_index]['points_3d'])
    this_pose[:, 0] -= ori_wcx
    this_pose[:, 1] -= ori_wcy
    
    keep = (
        (this_pose[:, 0] > world_ltrb_mean_cen[0])
        & (this_pose[:, 0] < world_ltrb_mean_cen[2])
        & (this_pose[:, 1] > world_ltrb_mean_cen[1])
        & (this_pose[:, 1] < world_ltrb_mean_cen[3])
    )
    this_pose = this_pose[keep]
    scatter_points_arr.append(this_pose)
    person_ID_list.append(str(poses_3d_all_timestamps[timestamp_to_plot][pose_index]['id']))
    ax.text(this_pose[-1,0], this_pose[-1,1], this_pose[-1,2]+10, f'ID: {person_ID_list[pose_index]}', color='black')
    
# Only do the scatter inside the loop
# top head is last keypoint 
scatter_points_arr = np.array(scatter_points_arr).reshape(-1,3)

ax.scatter(scatter_points_arr[:,0], scatter_points_arr[:,1], scatter_points_arr[:,2], c = 'g', marker='o')
# drawing updated values
plt.show()

# %%
plot_3d_animation = True
if plot_3d_animation:
    ori_wcx = np.mean(world_ltrb[0::2])
    ori_wcy = np.mean(world_ltrb[1::2])
    world_ltrb_mean_cen = world_ltrb.copy()
    world_ltrb_mean_cen[0::2] -= ori_wcx
    world_ltrb_mean_cen[1::2] -= ori_wcy
    camera_id_list = list(camera_id_list)

    # Set up the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_xlim3d(world_ltrb_mean_cen[0], world_ltrb_mean_cen[2])
    #ax.set_ylim3d(world_ltrb_mean_cen[1], world_ltrb_mean_cen[3])
    ax.set_xlim3d(-2000, 2000)
    ax.set_ylim3d(-1000, 1000)
    ax.set_zlim3d(0, 300)
        
    
    #show_cam_location_on_3d_plot(camera_3d_location, camera_look_at, magnitude=20, ax = ax)
    show_cam_location_on_3d_plot(camera_id_list, calibration, magnitude=20, ax = ax)
    scatter = ax.scatter([], [], [], c='g', marker='o')
    #person_ID_text = ax.text(0,0,0, s = '', color='black')
    
    # Animation update function
    def update(index):
        # Clear the previous frame
        scatter._offsets3d = ([], [], [])
        #person_ID_text.set_text('')
        #person_ID_text.set_position((0,0,0))
    
        timestamp_to_plot = list(poses_3d_all_timestamps.keys())[index]
        scatter_points_list = []
        # Get the 3D points for the current frame
        if all(value is not None for value in poses_3d_all_timestamps[timestamp_to_plot]):
            for pose_index in range(len(poses_3d_all_timestamps[timestamp_to_plot])):
                this_pose = np.array(poses_3d_all_timestamps[timestamp_to_plot][pose_index]['points_3d'])
                this_pose[:, 0] -= ori_wcx
                this_pose[:, 1] -= ori_wcy
                
                keep = (
                    (this_pose[:, 0] > world_ltrb_mean_cen[0])
                    & (this_pose[:, 0] < world_ltrb_mean_cen[2])
                    & (this_pose[:, 1] > world_ltrb_mean_cen[1])
                    & (this_pose[:, 1] < world_ltrb_mean_cen[3])
                )
                #this_pose = this_pose[keep]
                scatter_points_list.append(this_pose)
                person_ID = str(poses_3d_all_timestamps[timestamp_to_plot][pose_index]['id'])
                                
                #person_ID_text.set_text(f'ID: {person_ID}')
                #person_ID_text.set_position((this_pose[-1,0], this_pose[-1,1], this_pose[-1,2]+10))
                
            scatter_points_arr = np.array(scatter_points_list).reshape(-1,3)
            scatter._offsets3d = (scatter_points_arr[:,0], scatter_points_arr[:,1], scatter_points_arr[:,2])
        
        # Set the plot title with the timestamp
        ax.set_title(f'Timestamp: {timestamp_to_plot}')
    
    # Create the animation
    animation = FuncAnimation(fig, update, len(poses_3d_all_timestamps), interval=20, repeat=False)

    # Show the animation
    plt.show()


# %%



