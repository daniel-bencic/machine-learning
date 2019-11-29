import os
import numpy as np
import skimage.io as sio
import skimage.transform as strans

def get_people(base, dirs):
    names = []
    paths = []
    for d in dirs:
        path = os.path.join(base, d)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            if count >= 70:
                names.append(d)
                paths.append(path)
    return paths, names

def load_images(paths):
    imgs = []
    t_imgs = []
    for p in paths:
        p_imgs = sorted(os.listdir(p))
        l_imgs = []
        for p_img in p_imgs:
            img = sio.imread(os.path.join(p, p_img), as_gray=True)[90:210, 75:195]
            l_imgs.append(strans.resize(img, (32, 32)))
        t_imgs.append(l_imgs.pop())
        imgs.append(l_imgs)
    return imgs, t_imgs

def stack_image(img):
    stacked = np.array([])
    for row in img:
        stacked = np.concatenate((stacked, row))
    return stacked

def unstack_image(img, ppr):
    unstacked = []
    for i in range(int(img.shape[0] / ppr)):
        unstacked.append(img[i * ppr:(i + 1) * ppr])
    return np.array(unstacked)

