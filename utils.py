import cv2
import numpy as np


def drawInliers(img1, img2, inliers):
    w, h, c = img1.shape
    joined_image = np.concatenate((img1, img2), axis=1)
    img1_pts = inliers[:, 0:2].astype(int)
    img2_pts = inliers[:, 2:].astype(int)
    for i in range(img1_pts.shape[0]):
        pt_img1 = (img1_pts[i, 0], img1_pts[i, 1])
        pt_img2 = (h+img2_pts[i, 0], img2_pts[i, 1])
        color = tuple(np.random.randint(0,255,3).tolist())
        joined_image = cv2.circle(joined_image, pt_img1, radius=0, color=color, thickness=3)
        joined_image = cv2.circle(joined_image, pt_img2, radius=0, color=color, thickness=3)
        joined_image = cv2.line(joined_image, pt_img1, pt_img2, color=color, thickness=1)
        # if inlier_flags[i] == 0:
        #     joined_image = cv2.circle(joined_image, pt_img1, radius=0, color=(0, 0, 255), thickness=3)
        #     joined_image = cv2.circle(joined_image, pt_img2, radius=0, color=(0, 0, 255), thickness=3)
        #     joined_image = cv2.line(joined_image, pt_img1, pt_img2, color=(0, 0, 255), thickness=1)
        # if inlier_flags[i] == 1:
        #     joined_image = cv2.circle(joined_image, pt_img1, radius=0, color=(0, 255, 0), thickness=3)
        #     joined_image = cv2.circle(joined_image, pt_img2, radius=0, color=(0, 255, 0), thickness=3)
        #     joined_image = cv2.line(joined_image, pt_img1, pt_img2, color=(0, 255, 0), thickness=1)
    cv2.imshow("Inliers", joined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()