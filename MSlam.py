import cv2
from ORBfeatureMatcher import *
from FundamentalMatrix import *




def main():
    image1 = cv2.imread('data/1.png')
    image2 = cv2.imread('data/2.png')

    matches= ORBMatcher(image1, image2, no_kpts=50)
    inliers, best_F = getBestF(matches)
    print(inliers.shape)
    # print(matches.shape)

    pass



if __name__ == '__main__':
    main()