import numpy as np
import os
import argparse
from PIL import Image

# Constants that probably don't change
# TODO(roger): these constants are copied in several files and should be unified and documented
DEPTH_SCALING_FACTOR    = 1000.0
DEPTH_CUTOFF            = 1.0
VOXEL_SIZE              =0.005

# From handeye.ipynb
cam_to_gripper_rot = np.array([[-7.77766820e-02, -9.44073658e-01, -3.20430518e-01],
 [ 9.96970509e-01, -7.34019190e-02, -2.57286360e-02],
 [ 7.69512542e-04, -3.21460864e-01,  9.46922553e-01]])

cam_to_gripper_trans = np.array([[ 0.10598264], [-0.02439176], [-0.08322135]])

cam_to_gripper_pose = np.eye(4)
cam_to_gripper_pose[:3, :3] = cam_to_gripper_rot
cam_to_gripper_pose[:3, 3] = cam_to_gripper_trans.squeeze()

def gather_single_sequence_data(base_dir):
    assert os.path.exists(base_dir), f"Path {base_dir} does not exist"
    intrinsic_mat = np.load(os.path.join(base_dir, 'camera_intrinsics.npz'))

    fx, fy = intrinsic_mat['fx'], intrinsic_mat['fy']
    ppx, ppy = intrinsic_mat['ppx'], intrinsic_mat['ppy']

    depth_image_path = os.path.join(base_dir, 'camera_depth')
    assert os.path.exists(depth_image_path), f"Path {depth_image_path} does not exist"
    depth_image_path_list = sorted([os.path.join(depth_image_path, f) for f in os.listdir(depth_image_path) if f.endswith('.npy')])
    depth_image_list = [np.load(f) for f in depth_image_path_list]

    rgb_image_path = os.path.join(base_dir, 'camera_rgb')
    assert os.path.exists(rgb_image_path), f"Path {rgb_image_path} does not exist"
    rgb_image_path_list = sorted([os.path.join(rgb_image_path, f) for f in os.listdir(rgb_image_path) if f.endswith('.png')])
    rgb_image_list = [np.array(Image.open(f).convert('RGB')) for f in rgb_image_path_list]

    assert len(depth_image_list) == len(rgb_image_list)
    assert depth_image_list[0].shape == rgb_image_list[0].shape[:2]

    pose_path = os.path.join(base_dir, 'poses')
    assert os.path.exists(pose_path), f"Path {pose_path} does not exist"
    pose_path_list = sorted([os.path.join(pose_path, f) for f in os.listdir(pose_path) if f.endswith('.npy')])
    pose_list = [np.load(f) @ cam_to_gripper_pose for f in pose_path_list]

    return {
        'fx': fx,
        'fy': fy,
        'ppx': ppx,
        'ppy': ppy,
        'depth_image_list': depth_image_list,
        'rgb_image_list': rgb_image_list,
        'pose_list': pose_list
    }

def main(src_dir, dst_dir):
    dataset_dict = None

    for seq_name in os.listdir(src_dir):
        seq_dir = os.path.join(src_dir, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        print(f'Processing sequence {seq_name}')
        seq_data = gather_single_sequence_data(seq_dir)

        if dataset_dict is None:
            dataset_dict = seq_data
        else:
            assert np.isclose(dataset_dict['fx'], seq_data['fx'])
            assert np.isclose(dataset_dict['fy'], seq_data['fy'])
            assert np.isclose(dataset_dict['ppx'], seq_data['ppx'])
            assert np.isclose(dataset_dict['ppy'], seq_data['ppy'])
            assert dataset_dict['depth_image_list'][0].shape == seq_data['depth_image_list'][0].shape
            assert dataset_dict['rgb_image_list'][0].shape == seq_data['rgb_image_list'][0].shape
            dataset_dict['depth_image_list'] += seq_data['depth_image_list']
            dataset_dict['rgb_image_list'] += seq_data['rgb_image_list']
            dataset_dict['pose_list'] += seq_data['pose_list']
    
    os.makedirs(dst_dir, exist_ok=True)
    color_dir = os.path.join(dst_dir, 'color')
    depth_dir = os.path.join(dst_dir, 'depth')
    pose_dir = os.path.join(dst_dir, 'pose')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # Save intrinsics
    intrinsics_mat = np.array([
        [dataset_dict['fx'], 0, dataset_dict['ppx']],
        [0, dataset_dict['fy'], dataset_dict['ppy']],
        [0, 0, 1]
    ])

    np.savetxt(os.path.join(dst_dir, 'intrinsics.txt'), intrinsics_mat)

    # Save color images
    for i, img in enumerate(dataset_dict['rgb_image_list']):
        Image.fromarray(img).save(os.path.join(color_dir, f'{i:06d}.png'))
    
    # Save depth images
    for i, img in enumerate(dataset_dict['depth_image_list']):
        img = img.astype(np.uint16)
        Image.fromarray(img).save(os.path.join(depth_dir, f'{i:06d}.png'))
    
    # Save poses as txt
    for i, pose in enumerate(dataset_dict['pose_list']):
        np.savetxt(os.path.join(pose_dir, f'{i:06d}.txt'), pose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert dataset to plain format')
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing the dataset')
    parser.add_argument('--dst_dir', type=str, required=True, help='Destination directory to save the converted dataset')
    args = parser.parse_args()
    main(args.src_dir, args.dst_dir)
