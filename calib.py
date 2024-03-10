import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 5), 0.08, 0.06, aruco_dict)

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)        
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])              
        
        decimator += 1   

    imsize = gray.shape
    return allCorners, allIds, imsize

def calibrate_camera(allCorners, allIds, imsize, cameraMatrixInit):   
    """
    Calibrates the camera using the detected corners.
    """
    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO) 
    # flags = (cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients0, 
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics, 
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


if __name__ == "__main__":
    datadir = "./cameras/camera1_rgb/"
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png")])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    cameraMatrixInit = np.array([[1361.57, 0., 956.298],
                                [0., 1361.57, 546.581],
                                [0., 0., 1.]])
    allCorners, allIds, imsize = read_chessboards(images)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize, cameraMatrixInit)

    print("Camera matrix : \n {0}".format(mtx))
    print(dist)

    # check calibration
    i=5
    plt.figure()
    frame = cv2.imread(images[i])
    img_undist = cv2.undistort(frame,mtx,dist,None)
    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img_undist)
    plt.title("Corrected image")
    plt.axis("off")
    plt.show()