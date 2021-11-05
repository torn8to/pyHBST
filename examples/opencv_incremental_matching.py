import cv2
import pyhbst
import glob
import os
import numpy as np

def get_matchables_from_ocv(kpts, desc, img_id):
    matchables = []
    if desc.shape[1] == 32:
        for i in range(desc.shape[0]):
            ma = pyhbst.Matchable256(list(kpts[i].pt), desc[i,:].tolist(), img_id)
            matchables.append(ma)

    return matchables

detector = cv2.ORB_create(100)
split_strat = pyhbst.SplitEven
max_hamming_dist = 256
tree256 = pyhbst.BinarySearchTree256()

images = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/*.ppm'))

img_dict = {}
for idx, img_path in enumerate(images):
    I = cv2.imread(img_path, 0)
    img_dict[idx] = I
    kpts, desc = detector.detectAndCompute(I,None)

    matchables = get_matchables_from_ocv(kpts, desc, idx)

    #matches = None
    #if idx == 0:
    #    tree256.add(matchables, split_strat)
    #else:
        #matches = tree256.match(matchables, max_hamming_dist)
        #tree256.add(matchables, split_strat)
    matches = tree256.matchAndAdd(matchables, max_hamming_dist, split_strat)
    print(matches)
    # show matches
    #if matches:
        #cv2.imshow()

tree256.clear(False)