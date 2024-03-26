import os
import cv2
import numpy as np

def read_data(base_dir):
    rgb_folder = os.path.join(base_dir, 'rgb')
    depth_folder = os.path.join(base_dir, 'depth')
    pose_folder = os.path.join(base_dir, 'poses')

    # Read RGB images
    rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith('.png')]
    rgb_files.sort() 
    rgb_list = [cv2.imread(os.path.join(rgb_folder, f)) for f in rgb_files]

    # Read depth images
    depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.npy')]
    depth_files.sort() 
    depth_list = [np.load(os.path.join(depth_folder, f)) for f in depth_files]

    # Read poses
    pose_files = [f for f in os.listdir(pose_folder) if f.endswith('.npy')]
    pose_files.sort() 
    pose_list = [np.load(os.path.join(pose_folder, f)) for f in pose_files]

    assert len(rgb_list) == len(pose_list) == len(depth_list)

    # Load the intrinsic parameters
    camera_params = np.load(os.path.join(base_dir, 'camera_intrinsics.npz'))
    fx = camera_params['fx']
    fy = camera_params['fy']
    ppx = camera_params['ppx']
    ppy = camera_params['ppy']
    camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])

    return rgb_list, depth_list, pose_list, camera_matrix