import cv2
from ORBfeatureMatcher import *
from FundamentalMatrix import *
from utils import *




def main():
    image1 = cv2.imread('data/1.png')
    image2 = cv2.imread('data/2.png')

    matches = ORBMatcher(image1, image2, no_kpts=100, show_matches=True)
    inliers, F = getBestF(matches,1000,0.001)
    
    drawInliers(image1, image2, inliers)
    print("F: ", )
    print(F)
    K = np.array([[531.122155322710, 0, 407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])
    E = getE(K,F)
    R, t = extractCameraPose(E)
    triangulated_pts = triangulatePts(R,t,K,inliers)
    id, R_best, t_best = cheirality_check(triangulated_pts,R,t)

    pts_3d = triangulated_pts[id]
    pts_3d = pts_3d/pts_3d[3,:]

    pass



if __name__ == '__main__':
    main()
