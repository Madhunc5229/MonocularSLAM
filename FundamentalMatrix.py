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

def extractCameraPose(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    C= list()
    R= list()
    C.append(U[:,2])
    C.append(-U[:,2])
    C.append(U[:,2])
    C.append(-U[:,2])

    R.append(np.dot(U, np.dot(W,Vt)))
    R.append(np.dot(U, np.dot(W,Vt)))
    R.append(np.dot(U, np.dot(W.T,Vt)))
    R.append(np.dot(U, np.dot(W.T,Vt)))
    
    for i in range(4):
        if(np.linalg.det(R[i])<0):
            C[i] = -C[i]
            R[i] = -R[i]

    return R, C


def triangulatePts(R,t,K,inliers):

    I_3 = np.identity(3)
    R1 = np.identity(3)
    C1 = np.zeros((3,1))

    P1 = np.dot(K, np.dot(R1, np.hstack((I_3, -C1))))
    
    trinagluated_3d_pts = list()
    for i in range(len(R)):
        x1 = inliers[:,:2]
        x2 = inliers[:,2:]

        P2 = np.dot(K, np.dot(R[i], np.hstack((I_3, -t[i].reshape(3,1)))))

        triangulated_pts = cv2.triangulatePoints(P1, P2, x1.T, x2.T)

        trinagluated_3d_pts.append(triangulated_pts)
    
    return trinagluated_3d_pts


def positive_zCount(points, R1, C1):

    points = points.T[:,:-1]
    points_translated = points-C1.T
    r3 = R1[-1,:]
    z = np.dot(r3, points_translated.T)
    z_count = np.where(z>0)[0].size

    return z_count


def cheirality_check(points_3d, R, t):
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    positive_zc_1=[]
    positive_zc_2=[]
    for idx in range(len(points_3d)):
        points = points_3d[idx]
        points = points/points[3,:]
        z1 = positive_zCount(points, R1, C1)
        z2 = positive_zCount(points, R[idx], t[idx])

        positive_zc_1.append(z1)
        positive_zc_2.append(z2)

    positive_zc_1 = np.array(positive_zc_1)
    positive_zc_2 = np.array(positive_zc_2)
    threshold_points = int(points_3d[0].shape[1]//2)

    pose_idx = np.intersect1d(np.where(positive_zc_1>threshold_points), np.where(positive_zc_2>threshold_points))[0]

    R2 = R[pose_idx]
    C2 = t[pose_idx]

    return pose_idx, R2, C2