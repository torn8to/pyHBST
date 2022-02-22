# pyHBST
Python bindings for Hamming Binary Search Trees.

The Python bindings are for the HBST C++ implementation: https://gitlab.com/srrg-software/srrg_hbst   
The method is described in this paper:

    @article{2018-schlegel-hbst, 
      author  = {D. Schlegel and G. Grisetti}, 
      journal = {IEEE Robotics and Automation Letters}, 
      title   = {{HBST: A Hamming Distance Embedding Binary Search Tree for Feature-Based Visual Place Recognition}}, 
      year    = {2018}, 
      volume  = {3}, 
      number  = {4}, 
      pages   = {3741-3748}
    }

## Usage
### Examples
You can find two examples how to use the library in [examples](examples).
 * The first example [opencv_incremental_matching_orb.py](examples/opencv_incremental_matching_orb.py), depicts how the library can be used to incrementally add and match images to a binary search tree with 256bit sized descriptors. Instead of exhaustively matching images to each other, we find the best matching images with corresponding 2D points and descriptors from the search tree and only need to touch the descriptors from each image once. 
 * The seconds example [opencv_incremental_matching_akaze_binarization.py](examples/opencv_incremental_matching_akaze_binarization.py) depicts a smaller tree with only 64-bit and how to use binarized AKAZE float descriptors to achieve the same thing.

### General usage
``` Python
# construct a tree supporting some descriptor size (64,128,256,512)
tree256 = pyhbst.BinarySearchTree256() # descriptor size 256-bit
# get some keypoints and descriptors for that image
kpts, desc = orb.detectAndCompute(image,None)
kpts_list = [list(kpt.pt) for kpt in kpts]
# incrementally match and add this image
hamming_dist = 10
matches = tree.matchAndAdd(kpts_list, desc.tolist(), img_id, hamming_dist, pyhbst.SplitEven)

if matches:
    # iterate all matches in tree
    for m in tree_matches:
        match_obj = tree_matches[m]
        # if we have found a match with another image
        if match_obj:
            nr_matches = len(match_obj)
            for i in range(nr_matches):
                query_kp = match_obj[i].object_query
                refer_kp = match_obj[i].object_references[0]

# important! clean up
tree256.clear(True)
``` 


## Installation
### From PyPi for Linux
This wheel does however not support architecture aware optimizations which can yield significant speedups (see below).
``` bash
pip install pyhbst
```

### Building on your own
Using -DBUILD_WITH_MARCH_NATIVE=ON will significantly increase performance.
You can enable it by setting an env variable BUILD_MARCH_NATIVE=1.

``` bash
git clone --recursive https://github.com/urbste/pyHBST
cd pyHBST
BUILD_MARCH_NATIVE=1 python setup.py bdist_wheel
cd dist && pip install *.whl
```

