import pickle as pkl
import numpy as np
import cv2,sys
sys.setrecursionlimit(1000000)

def find_neighbours(targets, pixels):
    close = set()
    for t in targets:
        close.update(set([(t[0],t[1]+1),(t[0],t[1]-1),
                        (t[0]+1,t[1]),(t[0]-1,t[1])]))
    neigh = set.intersection(close,pixels)
    if len(neigh) > 0:
        neigh.update(find_neighbours(neigh,pixels-neigh))
        neigh.update(targets)
        return neigh
    else:
        return targets


def find_components(pixels):
    if len(pixels) == 0:
        return []
    target = list(pixels)[0]
    pixels.remove(target)
    neigh = find_neighbours(set([target]),pixels)
    print("found component")
    return [list(neigh)] + find_components(pixels - neigh)



'''
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
'''



def find(classes,probabilities,img_shape,filter_size,stride):
    print("Finding Boxes...")

    i = 0
    k = 0.70
    probab_threshold = 0.93

    #hit_map = np.ones(img_shape)
    probab_map = np.zeros(img_shape)
    for y in range(filter_size//2,img_shape[1]-filter_size//2, stride):
        ymin, ymax = y - int(k*stride), y + int(k*stride)
        for x in range(filter_size//2,img_shape[0]-filter_size//2, stride):
            xmin, xmax = x - int(k*stride), x + int(k*stride)
            if probabilities[i][1] >= probab_threshold:
                probab_map[xmin:xmax,ymin:ymax] += probabilities[i][1]
                #hit_map[xmin:xmax,ymin:ymax] += 1
            i += 1

    probab_map[probab_map<1] = 0
    indices = np.where(probab_map > 0)
    pixels = [(indices[0][i],indices[1][i]) for i in range(len(indices[0]))]

    components = find_components(set(pixels))
    boxes = []
    for compId, compVal in enumerate(components):
        comp_x = [coord[1] for coord in compVal]
        comp_y = [coord[0] for coord in compVal]
        comp_xmax, comp_xmin = max(comp_x), min(comp_x)
        comp_ymax, comp_ymin = max(comp_y), min(comp_y)
        if (comp_xmax-comp_xmin) * (comp_ymax-comp_ymin) > (k*stride)**2:
            boxes.append((comp_xmax,comp_xmin,comp_ymax,comp_ymin))

    #cv2.imshow("probability map",probab_map)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return boxes



if __name__ == "__main__":
    #find(0,0,0,0,0)
    print(find_components(set([(0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(4,4),(5,4)])))
