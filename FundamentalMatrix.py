import numpy as np
import cv2
from ORBfeatureMatcher import ORBMatcher

def normalizeCorrespondences(img_pts):
    x = img_pts[:, 0]
    y = img_pts[:, 1]
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    x_new = x - centroid_x
    y_new = y - centroid_y
    mean_distance = np.mean(np.sqrt(x_new**2 + y_new**2))
    scale = np.sqrt(2)/mean_distance
    
    transform_mtrx = np.eye(3)
    transform_mtrx[0, 0] = scale
    transform_mtrx[0, 2] = -scale*centroid_x
    transform_mtrx[1, 1] = scale
    transform_mtrx[1, 2] = -scale*centroid_y 
    
    hmg_pts = np.hstack((img_pts, np.ones((len(img_pts), 1))))
    normalized_pts = (transform_mtrx.dot(hmg_pts.T)).T

    return normalized_pts, transform_mtrx


def getF(sample_pts):

    x, T = normalizeCorrespondences(sample_pts[:,:2])
    x_p, T_p = normalizeCorrespondences(sample_pts[:,2:])
    A_matrix = []
    for i in range(sample_pts.shape[0]):
        u, v = x[i, 0], x[i, 1]
        u_p, v_p = x_p[i, 0], x_p[i, 1]
        A_row = [u*u_p, u_p*v, u_p, u*v_p, v*v_p, v_p, u, v, 1]
        A_matrix.append(A_row)
    A_matrix = np.array(A_matrix)
    [U, S, V_T] = np.linalg.svd(A_matrix)
    F_elems = V_T.T[:, -1]
    F = F_elems.reshape((3, 3))

    u, s, v_T = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = np.dot(u, np.dot(s, v_T))
    
    F = np.dot(T_p.T, np.dot(F, T))
    F = F / F[2, 2]
    return F

def getBestF(matches, iterations = 1000, thresh = 0.01):
    max_inliers = 0
    best_F = None

    best_inliers_1 = list()
    best_inliers_2 = list()

    for i in range(iterations):
        sample_pts = matches[np.random.choice(matches.shape[0],size=8),:]

        F = getF(sample_pts)
        
        inliers_1 = list()
        inliers_2 = list()

        for i in range(matches.shape[0]):
            pt = matches[i]
            x1 = np.array([pt[0], pt[1],1])
            x2 = np.array([pt[2], pt[3],1])
            error = np.abs(np.dot(x1.T, np.dot(F,x2)))

            if error < thresh:
                inliers_1.append(x1)
                inliers_2.append(x2)
            
        num_of_inliers = len(inliers_1)

        if num_of_inliers > max_inliers:
            best_inliers_1 = inliers_1
            best_inliers_2 = inliers_2
            best_F = F
            max_inliers = len(best_inliers_1)

    inliers = np.hstack((np.array(best_inliers_1)[:,:2], np.array(best_inliers_2)[:,:2]))

    return inliers, best_F

def getE(K,F):
    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)
    S = np.diag(S)
    S[2, 2] = 0
    E = np.dot(U, np.dot(S, V_T))
    return E
    
