import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def show_cam_image_with_pose_keypoints(frame_cam_0, frame_cam_0_pose, image_wh = [360, 288]):
    """
    Visualize camera image with pose keypoints overlaid
    
    Args:
        frame_cam_0: Dictionary containing camera frame image
        frame_cam_0_pose: Dictionary containing pose keypoints
        image_wh: Image width and height
    """
    # Create a figure. Equal aspect so circles look circular
    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(frame_cam_0['image'])

    # plot poses
    if frame_cam_0_pose is not None:
        point_radius = 2
        score_thresh = 0.1
        rendering = frame_cam_0['image'].copy()
        render_h, render_w = rendering.shape[0:2]
        scale_x = render_w / image_wh[0]
        scale_y = render_h / image_wh[1]
        for i, pose in enumerate(frame_cam_0_pose["poses"]):
            raw_points = np.copy(pose["points_2d"])
            keypoints_score = pose["scores"]
            raw_points[:, 0] *= scale_x
            raw_points[:, 1] *= scale_y
            keypoints = [tuple(map(int, point)) for point in raw_points]
            for k, (point, score) in enumerate(zip(keypoints, keypoints_score)):
                if score < score_thresh:
                    continue
                circ = Circle(point,point_radius)
                ax.add_patch(circ)
    
    plt.show() 

def show_cam_location_on_3d_plot(camera_id_list, calibration, magnitude = 5, ax = None):
    """
    Visualize camera locations and orientations in 3D space
    
    Args:
        camera_id_list: List of camera IDs
        calibration: Calibration object containing camera parameters
        magnitude: Size of the camera pyramid visualization
        ax: Optional matplotlib 3D axis to plot on
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
     
    for cam_id in camera_id_list:
        location = calibration.cameras[cam_id].location
        look_at_point = calibration.cameras[cam_id].look_at
        ax.scatter(location[0], location[1], location[2], c='red', marker='o')
        ax.text(location[0], location[1], location[2], f'{cam_id}', color='black')

        axis_vector = (np.array(look_at_point) - np.array(location))/5

        pyramid_vertices = [
            location,
            (location[0] + axis_vector[0], location[1] + axis_vector[1], location[2] + axis_vector[2]),
            (location[0] + axis_vector[0], location[1] + axis_vector[1], location[2] + axis_vector[2]),
            (location[0] + axis_vector[0] -  magnitude, location[1] + axis_vector[1] -  magnitude, location[2] + axis_vector[2] -  magnitude),
            (location[0] + axis_vector[0] +  magnitude, location[1] + axis_vector[1] -  magnitude, location[2] + axis_vector[2] -  magnitude),
            (location[0] + axis_vector[0], location[1] + axis_vector[1], location[2] + axis_vector[2])
        ]

        pyramid_faces = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 1, 3],
            [0, 4, 1],
            [2, 1, 3],
            [2, 1, 4],
            [5, 2, 3],
            [5, 2, 4]
        ]

        pyramid_vertices = np.array(pyramid_vertices)
        for face in pyramid_faces:
            ax.plot(pyramid_vertices[face, 0], pyramid_vertices[face, 1], pyramid_vertices[face, 2], c='blue') 