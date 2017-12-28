import pickle as pkl
import numpy as np
import cv2,sys
sys.setrecursionlimit(1000000)

def find_conn_components(disconn):
    if not disconn:
        return []
    idx1 = 0
    val1 = disconn[0]
    for p1 in val1:
        neigh = set([(p1[0],p1[1]+1),(p1[0],p1[1]-1),(p1[0]+1,p1[1]),(p1[0]-1,p1[1])])
        for idx2, val2 in enumerate(disconn[1:]):
            for p2 in val2:
                if p2 in neigh:
                    conn = disconn[:]
                    del conn[idx1]
                    del conn[idx2]
                    conn.append(set.union(val1,val2))
                    return find_conn_components(conn)
    print("found box")
    return [disconn[0]] + find_conn_components(disconn[1:])

def find(classes,probabilities,img_shape,filter_size,stride):

    probab_map = np.zeros(img_shape)
    #hit_map = np.ones(img_shape)
    i = 0
    for y in range(filter_size//2,img_shape[0]-filter_size//2, stride):
        ymin, ymax = y - stride//2, y + stride//2
        for x in range(filter_size//2,img_shape[1]-filter_size//2, stride):
            xmin, xmax = x - stride//2, x + stride//2
            if probabilities[i][1] >= 0.85:
                probab_map[ymin:ymax,xmin:xmax] += probabilities[i][1]
                #hit_map[xmin:xmax,ymin:ymax] += 1
            i += 1

    probab_map[probab_map<0.97] = 0
    indices = np.where(probab_map > 0)
    pixels = [set([(indices[0][i],indices[1][i])]) for i in range(len(indices[0]))]
    boxes = []

    for comp in find_conn_components(pixels):
        comp_x = [coord[0] for coord in comp]
        comp_y = [coord[1] for coord in comp]
        comp_xmax, comp_xmin = max(comp_x), min(comp_x)
        comp_ymax, comp_ymin = max(comp_y), min(comp_y)
        boxes.append((comp_xmax,comp_xmin,comp_ymax,comp_ymin))

    return boxes



if __name__ == "__main__":
    find(0,0,0,0,0)
