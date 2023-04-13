import numpy as np
import cv2

def ORBMatcher(image1, image2, no_kpts=None, show_matches=True):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    final_matches = matcher.match(des1, des2, None)

    final_matches = sorted(final_matches, key = lambda x:x.distance)

    final_matches = final_matches[:no_kpts]

    matches = list()

    for match in final_matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        matches.append([pt1[0], pt1[1], pt2[0], pt2[1]])

    matches = np.array(matches)

    if show_matches:
        fram = cv2.drawMatches(image1, kp1, image2, kp2, final_matches, None)
        cv2.imshow('Feature Matching', fram)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return matches


