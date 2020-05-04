import os
import random
import cv2
import copy
import re
import numpy as np


def rotate(img):
    h, w, l = img.shape
    bounds = [-20, 20]  # bounds for the rotation angle in degrees
    r = random.randint(bounds[0], bounds[1])
    rmatr = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=r, scale=1)
    img = cv2.warpAffine(img, rmatr, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def shift(img):
    h, w, l = img.shape
    bounds = [-25, 25]  # bounds for the shift in x and y in pixels
    sx = random.randint(bounds[0], bounds[1])
    sy = random.randint(bounds[0], bounds[1])
    smatr = np.float32([[1, 0, sx], [0, 1, sy]])
    img = cv2.warpAffine(img, smatr, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def zoomin(img):
    h, w, l = img.shape
    bounds = [0, 30]  # bounds for the zoom factor in %
    zx = 1 + random.randint(bounds[0], bounds[1]) / 100
    zy = 1 + random.randint(bounds[0], bounds[1]) / 100
    inc_w = round((w * zx) - w)
    inc_h = round((h * zy) - h)
    img = cv2.resize(img, (w + inc_w, h + inc_h), interpolation=cv2.INTER_AREA)
    img = img[inc_h // 2:inc_h // 2 + h, inc_w // 2:inc_w // 2 + w]
    return img


def noise(img):
    h, w, l = img.shape
    bounds = [0, 0.75]
    s2 = random.uniform(bounds[0], bounds[1])
    gauss = np.random.normal(0, s2, img.shape).astype('uint8')
    # gauss = gauss.reshape(h, w, l ).astype('uint8')
    # Add the Gaussian noise to the image
    img = cv2.add(img, gauss)
    return img


def blur(img):
    bounds = [1, 4]  # bounds for the kernel size
    k = random.randint(bounds[0], bounds[1])*2+1
    img = cv2.medianBlur(img, k, cv2.BORDER_REPLICATE)
    return img


def main():
    nb_images_to_generate = 100

    orig_path = '../data/originals/'
    dest_path = '../data/augmented/'

    if len(os.listdir(dest_path)) > 0:
        maxid = max([int(re.search("_[0-9]+", i).group()[1:]) for i in os.listdir(dest_path)])
    else:
        maxid = 1

    images_list = os.listdir(orig_path)

    available_transformations = {'rotate': rotate, 'shift': shift, 'zoom': zoomin, 'noise': noise, 'blur': blur}

    for nb in range(0, nb_images_to_generate):
        # select a random image
        image_file = random.choice(images_list)
        fname, ext = os.path.splitext(image_file)
        # image = cv2.imread(os.path.join(orig_path, image_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(os.path.join(orig_path, image_file))

        # nb of transformation to apply
        nb_transform = random.randint(1, len(available_transformations))

        for t in range(0, nb_transform):
            t_select = random.choice(list(available_transformations))
            img = available_transformations[t_select](image)

        # resize
        # img = cv2.resize(img, (50, 50))

        # save new image
        # img = ~img
        new_file_name = "from_%s_%d%s" % (fname, maxid, ext)
        cv2.imwrite(os.path.join(dest_path, new_file_name), img)
        maxid += 1


if __name__ == "__main__":
    main()
