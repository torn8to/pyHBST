import time

def get_matchables_from_ocv(tree, kpts, desc, img_id, h_dist, split_type):
    kpts_list = []
    for i in range(desc.shape[0]):
        kpts_list.append(list(kpts[i].pt))

    # performing match and add
    start = time.time()
    matches = tree.matchAndAdd(kpts_list, desc.tolist(), img_id, h_dist, split_type)
    print("Time for matchAndAdd: {}s".format(time.time()-start))
    return matches