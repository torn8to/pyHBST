import cv2
import pyhbst
import glob
import os
import numpy as np
import time
from utils import get_matchables_from_ocv

detector = cv2.ORB_create(1000)
max_hamming_dist = 50
hbst = pyhbst.BinarySearchTree256()

images = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/*.ppm'))

if not os.path.exists("debug_out"):
    os.makedirs("debug_out")

img_dict = {}
for idx, img_path in enumerate(images):
    I = cv2.imread(img_path)
    plot_shift_x = I.shape[1]
    img_dict[idx] = I
    kpts, desc = detector.detectAndCompute(I,None)
    print("Extracted {} keypoints from {}".format(desc.shape[0], idx))
    tree_matches = get_matchables_from_ocv(
            hbst, kpts, desc, idx, max_hamming_dist, pyhbst.SplitEven)

    # show matches
    if tree_matches:
        for m in tree_matches:
            match_obj = tree_matches[m]
            print("Putative matches between {} and {}: {}".format(idx, m, len(match_obj)))
            
            if match_obj:
                if len(match_obj) < 10:
                    continue
                conc_img = np.concatenate([I, img_dict[m]],1)
                nr_matches = len(match_obj)
                src_pts, dst_pts = [], []
                for i in range(nr_matches):
                    left_pt = match_obj[i].object_query
                    right_pt =  match_obj[i].object_references[0]

                    conc_img = cv2.line(conc_img, 
                        (int(left_pt[0]), int(left_pt[1])),
                        (int(right_pt[0]+plot_shift_x), int(right_pt[1])), (0,255,0), 2)

                    src_pts.append(left_pt)
                    dst_pts.append(right_pt)

                cv2.imwrite("debug_out/putative_matches"+str(idx)+"_"+str(m)+".jpg",conc_img)

                src_pts = np.float32(src_pts).reshape(-1,1,2)
                dst_pts = np.float32(dst_pts).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
                matchesMask = mask.ravel().tolist()
                print("Homography matches between {} and {}: {}".format(idx, m, np.sum(matchesMask)))

                conc_img = np.concatenate([I, img_dict[m]],1)
                for i, match in enumerate(matchesMask):
                    if match:
                        left_pt = src_pts[i].squeeze()
                        right_pt = dst_pts[i].squeeze()
                        conc_img = cv2.line(conc_img, 
                            (int(left_pt[0]), int(left_pt[1])),
                            (int(right_pt[0]+plot_shift_x), int(right_pt[1])), (0,255,0), 2) 
                cv2.imwrite("debug_out/homog_matches"+str(idx)+"_"+str(m)+".jpg",conc_img)


hbst.clear(True)