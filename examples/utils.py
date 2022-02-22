import time

def get_matchables_from_ocv(tree, kpts, desc, img_id, h_dist, split_type):
    kpts_list = [list(kpt.pt) for kpt in kpts]
    # performing match and add
    start = time.time()
    matches = tree.matchAndAdd(kpts_list, desc.tolist(), img_id, h_dist, split_type)
    tims_ms = 1000.*(time.time()-start)
    print("Time for matchAndAdd: {:.2f}ms".format(tims_ms))
    return matches