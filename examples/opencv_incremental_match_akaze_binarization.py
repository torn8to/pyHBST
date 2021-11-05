import cv2
import pyhbst
import glob
import os
import numpy as np
import time
from utils import get_matchables_from_ocv

detector = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_KAZE, 0, 3, 0.005)
max_hamming_dist = 5
tree64 = pyhbst.BinarySearchTree64()

images = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/*.ppm'))

img_dict = {}
for idx, img_path in enumerate(images):
    I = cv2.imread(img_path)
    plot_shift_x = I.shape[1]
    img_dict[idx] = I
    kpts, desc = detector.detectAndCompute(I,None)
    # binarize akaze descriptors to bool
    desc_bool = desc > 0
    tree_matches = get_matchables_from_ocv(
            tree64, kpts, desc_bool, idx, max_hamming_dist, pyhbst.SplitEven)

    # show matches
    if tree_matches:
        for m in tree_matches:
            match_obj = tree_matches[m]
            print("Matches between {} and {}: {}".format(idx, m, len(match_obj)))
            if match_obj:
                conc_img = np.concatenate([I, img_dict[m]],1)
                nr_matches = len(match_obj)
                for i in range(nr_matches):
                    left_pt = match_obj[i].object_query
                    right_pt =  match_obj[i].object_references[0]
                    conc_img = cv2.line(conc_img, 
                        (int(left_pt[0]), int(left_pt[1])),
                        (int(right_pt[0]+plot_shift_x), int(right_pt[1])), (0,255,0), 2)
                cv2.imshow("matches",conc_img)
                cv2.waitKey(0)

tree64.clear(True)