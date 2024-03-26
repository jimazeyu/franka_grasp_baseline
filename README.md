# tabletop_manipulation_calib
Table top manipulation calibration between the robot arm, the fixed cameras and the camera in hand.

## Install
python=3.10

## Usage

1. Use shoot_for_calib.ipynb to take photos and for calibration.
2. Use shoot_for_reconstruct.ipynb to take photos for reconstruction.
3. Use extrinsic_calib.ipynb to get the intrinsic matrix.
4. Use extrainsic_calib.ipynb to get the relative pose and do 3d reconstruction.(Only simple overlap now with manual scale)



utils:
- shoot
- 