import pathlib
import cv2
import numpy as np

from challenge.dataset.dataset import ChallengeVideo

HEIGHT = 874
WIDTH = 1164
F = 910

K = np.array([
    [910, 0, WIDTH/2],
    [0, 910, HEIGHT/2],
    [0, 0, 1]
])

c = ChallengeVideo(pathlib.Path("calib_challenge/labeled/1"))

sift = cv2.SIFT_create()

bf = cv2.BFMatcher()

def draw_path(R, t, img):
    rotV, _ = cv2.Rodrigues(R)

    points = np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0]], dtype=np.float32)

    projectedPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))

    img = cv2.line(img, projectedPoints[0].flatten().astype(np.int32), projectedPoints[1].flatten().astype(np.int32), (255,0,0), 3)
    img = cv2.line(img, projectedPoints[0].flatten().astype(np.int32), projectedPoints[2].flatten().astype(np.int32), (0,255,0), 3)
    img = cv2.line(img, projectedPoints[0].flatten().astype(np.int32), projectedPoints[3].flatten().astype(np.int32), (0,0,255), 3)

    return img

for i in range(len(c)-1):
    img1 = c[i][0]
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    mask = np.ones_like(img1_gray)
    mask[HEIGHT-300:HEIGHT] = 0 # cut out the car

    kp1, des1 = sift.detectAndCompute(img1_gray, mask)

    j = 1

    img2 = c[i+j][0]
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    kp2, des2 = sift.detectAndCompute(img2_gray, mask)

    matches = bf.knnMatch(des1,des2,k=2)

    ratio = 0.5

    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    
    matchesImage = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
    
    filtered_pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    filtered_pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    E, mask = cv2.findEssentialMat(filtered_pts1, filtered_pts2, K)
    
    points, R_est, t_est, mask_pose = cv2.recoverPose(E, filtered_pts1, filtered_pts2, K)

    path = draw_path(R_est, t_est, img1)

    print(np.linalg.norm(t_est))

    cv2.imshow("Path", path)
    cv2.imshow("Matches", matchesImage)
    cv2.waitKey(1)