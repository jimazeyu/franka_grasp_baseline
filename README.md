# tabletop_manipulation_calib
Table top manipulation calibration between the robot arm, the fixed cameras and the camera in hand.

1. Use shoot.ipynb to take photos and save pictures.(The code should be modified to save original depth image)
2. Use calib.py to get the intrinsic matrix.
3. Use extrainsic_calib.ipynb to get the relative pose and do 3d reconstruction.(Only simple overlap now with manual scale)
4. python=3.10