import cv2
from ORBfeatureMatcher import *
from FundamentalMatrix import *
from utils import *




def main():
    image1 = cv2.imread('data/1.png')
    image2 = cv2.imread('data/2.png')

    matches = ORBMatcher(image1, image2, no_kpts=100)
    inliers, F = getBestF(matches,1000,0.001)
    drawInliers(image1, image2, inliers)
    # print("F: ", )
    # print(F)
    # K = np.array([[531.122155322710, 0, 407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])
    # E = getE(K,F)

    pass



if __name__ == '__main__':
    main()