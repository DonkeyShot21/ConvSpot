
import pickle as pkl
import numpy as np
import cv2


def find(classes,probabilities,img_shape,filter_size,stride):
    prediction = pkl.load(open("./model/test_prob.pkl","rb"))
    classes = prediction["classes"]
    probabilities = prediction["probabilities"]
    img_shape = prediction["img_shape"]
    stride = prediction["stride"]
    filter_size = 28

    sunspot_map = np.zeros(img_shape)
    probab_map = np.zeros(img_shape)
    hit_map = np.zeros(img_shape)
    i = 0
    for ymin in range(0,img_shape[0]-filter_size, stride):
        ymax = ymin + filter_size
        for xmin in range(0,img_shape[1]-filter_size, stride):
            xmax = xmin + filter_size
            if probabilities[i][1] >= 0.6:
                sunspot_map[xmin:xmax,ymin:ymax] += 1
            probab_map[xmin:xmax,ymin:ymax] += probabilities[i][1]
            hit_map[xmin:xmax,ymin:ymax] += 1
            i += 1
    '''
    hit_map[hit_map == 0] = 1000 # to avoid division by 0
    sunspot_map = np.divide(sunspot_map,hit_map)
    sunspot_map = 255 * sunspot_map / (np.amax(sunspot_map)-np.amin(sunspot_map))
    sunspot_map = np.asarray(sunspot_map,dtype=np.uint8)
    '''

    hit_map[hit_map == 0] = 1000 # to avoid division by 0
    probab_map = np.divide(probab_map,hit_map)
    probab_map = 255 * probab_map / (np.amax(probab_map)-np.amin(probab_map))
    probab_map = np.asarray(probab_map,dtype=np.uint8)

    cv2.imwrite("probability_map.png",probab_map)

    probab_map[probab_map > 0.5] = 255

    cv2.imshow("sunspot map",sunspot_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return










if __name__ == "__main__":
    find(0,0,0,0,0)
