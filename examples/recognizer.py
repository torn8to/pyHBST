import cv2
import pyhbst

tree = pyhbst.BinaryTree256()
split_strat = pyhbst.SplitEven

desc1 = [True]*256
kp = [100,100]
img_id = 0
matchable1 = pyhbst.BinaryMatchable256(kp, desc1, img_id)
tree.add([matchable1], split_strat)

desc2 = [True]*256
desc2[0] = False
kp2 = [101, 101]
img_id = 1
matchable2 = pyhbst.BinaryMatchable256(kp2, desc2, img_id)

# no match
tree.matchAndAdd([matchable2], 1, split_strat)

# just match, should return one match
tree.match([matchable2], 2)


tree.clear(False)