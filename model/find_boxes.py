import pickle as pkl
import numpy as np
import cv2, sys, pickle
import scipy.ndimage.filters
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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
    return [list(neigh)] + find_components(pixels - neigh)

def boxes(classes,probabilities,img_shape,filter_size,stride):
    print("Finding Boxes...")

    i = 0
    k = 0.75
    #min_consensus = 0.3 # SOHO
    min_consensus = 0.4 # HELIOS
    #probab_threshold = 0.75 # SOHO
    probab_threshold = 0.95 # HELIOS

    hit_map = np.ones(img_shape)
    probab_map = np.zeros(img_shape)
    for y in range(filter_size//2,img_shape[1]-filter_size//2, stride):
        ymin, ymax = y - int(k*stride), y + int(k*stride)
        for x in range(filter_size//2,img_shape[0]-filter_size//2, stride):
            xmin, xmax = x - int(k*stride), x + int(k*stride)
            hit_map[xmin:xmax,ymin:ymax] += 1
            if probabilities[i][1] >= probab_threshold:
                probab_map[xmin:xmax,ymin:ymax] += probabilities[i][1]
            i += 1

    probab_map[probab_map < min_consensus * np.amax(hit_map)] = 0
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

    return boxes

def boxes_to_sunspots(img,boxes):
    for box in boxes:
        xmax, xmin, ymax, ymin = box
        group = img[ymin:ymax,xmin:xmax]
        original_shape = group.shape

        hist = cv2.calcHist([group],[0],None,[256],[0,256])
        plt.hist(group.ravel(),256,[0,256])
        plt.title('Histogram')
        plt.show()

        hmin = np.amin(group)
        hmax = np.amax(group)

        # K-Means
        init = np.array([hmin,(hmin+hmax)/2,hmax]).reshape(-1,1)
        km = KMeans(n_clusters=3,init=init)
        km.fit(group.reshape(-1,1))
        clustered = np.array(km.labels_,dtype=np.uint8).reshape(original_shape)

        # Gaussian Mixture Model
        # gmix = GaussianMixture(n_components=3, covariance_type='full')
        # gmix.fit(group.reshape(-1,1))

        indices = np.where(clustered == 0)
        pixels = [(indices[0][i],indices[1][i]) for i in range(len(indices[0]))]
        sunspots = find_components(set(pixels))
        print("Number of Sunspots",len(sunspots))

        z = 5
        cv2.imshow("original",cv2.resize(group,
            (z*group.shape[1], z*group.shape[0]),
            interpolation = cv2.INTER_NEAREST))
        cv2.imshow("detected",cv2.resize(clustered*100,
            (z*clustered.shape[1], z*clustered.shape[0]),
            interpolation = cv2.INTER_NEAREST))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = pkl.load(open("boxes.pkl","rb"))
    for k,v in obj.items():
        print(k)
        print(boxes_to_sunspots(v[0],v[1]))
