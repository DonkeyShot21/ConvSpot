import numpy as np
import xml.etree.ElementTree as ET
import os, sys, cv2, random

def find_sunspots(path):
    sunspots = {}
    for fn in [fn for fn in os.listdir(path) if ".xml" in fn]:
        tree = ET.parse(os.path.join(path,fn))
        root = tree.getroot()
        img_fn = fn.replace(".xml","")
        sunspots[img_fn] = []
        for ss in root.findall("object"):
            coord = []
            for box in ss.find("bndbox"):
                coord.append(int(box.text))
            sunspots[img_fn].append(coord)
    return sunspots

def compute_avg_size(sunspots):
    # ORDER: xmin ymin xmax ymax
    sizes = []
    for img, ss_list in sunspots.items():
        for ss in ss_list:
            sizes.append(ss[2]-ss[0])
            sizes.append(ss[3]-ss[1])
    return sum(sizes)/len(sizes)

def compute_area(a,b):
    # ORDER: xmin ymin xmax ymax
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy

def compute_label(slice_coord,ss_list):
    accept_tresh = 0.75
    label = 0
    slice_area = (slice_coord[2] - slice_coord[0]) * \
                 (slice_coord[3] - slice_coord[1])
    for ss in ss_list:
        intersect = compute_area(slice_coord,ss)
        if intersect is None:
            pass
        elif intersect / slice_area > accept_tresh or \
             intersect / ((ss[2]-ss[0])*(ss[3]-ss[1])) > accept_tresh:
             label = 1
    return label

def random_rotate(img):
    r = random.randint(0,4)
    return np.rot90(img,r)

def slice_and_label(img,ss_list,skip_ratio=1000,stride=10,size=28):
    slices, labels = [], []
    for xmin in range(0,len(img[0])-size,stride):
        xmax = xmin + size
        for ymin in range(0,len(img)-size,stride):
            ymax = ymin + size
            lab = compute_label([xmin,ymin,xmax,ymax],ss_list)
            if lab == 1 or random.randint(0,skip_ratio) % skip_ratio == 0:
                labels.append(lab)
                sl = img[ymin:ymax,xmin:xmax]
                sl = random_rotate(sl)
                slices.append(sl)
    return slices, labels

def slice_for_prediction(img, stride=7, size=28):
    disk_coord = {tuple(coords) for coords in np.argwhere(img > 50)}

    slices = []
    for xmin in range(0,len(img[0])-size,stride):
        xmax = xmin + size
        for ymin in range(0,len(img)-size,stride):
            ymax = ymin + size
            if disk_coord >= {(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)}:
                slices.append([[ymin,ymax,xmin,xmax],img[ymin:ymax,xmin:xmax]])
    return slices


def build_img_set(path):
    dataset = {"slices" : [], "labels" : []}
    sunspots = find_sunspots(path)
    for img_fn,ss_list in sunspots.items():
        print("slicing image:",img_fn)
        img_fn = os.path.join(path,img_fn+".jpg")
        img = cv2.imread(img_fn,0)
        print(img.shape)
        slices, labels = slice_and_label(img,ss_list,1000)
        dataset["slices"].extend(slices)
        dataset["labels"].extend(labels)
    return dataset


if __name__ == "__main__":
    path = sys.argv[1]

    ds = build_img_set(path)
    for l in set(ds["labels"]):
        print("label",l,"count:",len([x for x in ds["labels"] if x == l]))
