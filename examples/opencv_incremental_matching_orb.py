import cv2
import pyhbst
import glob
import os
import numpy as np
from utils import get_matchables_from_ocv

detector = cv2.ORB_create(1000)
max_hamming_dist = 30
tree256 = pyhbst.BinarySearchTree256()

images = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/*.ppm'))

img_dict = {}
for idx, img_path in enumerate(images):
    I = cv2.imread(img_path)
    plot_shift_x = I.shape[1]
    img_dict[idx] = I
    kpts, desc = detector.detectAndCompute(I,None)

    tree_matches = get_matchables_from_ocv(tree256, 
        kpts, desc, idx, max_hamming_dist, pyhbst.SplitEven)

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

tree256.clear(True)