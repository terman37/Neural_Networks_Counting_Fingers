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
    bounds = [0, 20]  # bounds for the zoom factor in %
    zx = 1 + random.randint(bounds[0], bounds[1]) / 100
    zy = 1 + random.randint(bounds[0], bounds[1]) / 100
    inc_w = round((w * zx) - w)
    inc_h = round((h * zy) - h)
    img = cv2.resize(img, (w + inc_w, h + inc_h), interpolation=cv2.INTER_AREA)
    img = img[inc_h // 2:inc_h // 2 + h, inc_w // 2:inc_w // 2 + w]
    return img


def noise(img):
    h, w, l = img.shape
    bounds = [0, 0.5]
    s2 = random.uniform(bounds[0], bounds[1])
    gauss = np.random.normal(0, s2, img.shape).astype('uint8')
    # gauss = gauss.reshape(h, w, l ).astype('uint8')
    # Add the Gaussian noise to the image
    img = cv2.add(img, gauss)
    return img


def blur(img):
    bounds = [1, 4]  # bounds for the kernel size
    k = random.randint(bounds[0], bounds[1]) * 2 + 1
    img = cv2.medianBlur(img, k, cv2.BORDER_REPLICATE)
    return img


def shear(img):
    h, w, l = img.shape
    shear_range = 20
    pts1 = np.float32([[5, 5], [100, 5], [5, 100]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 100 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, shear_M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .5 + np.random.uniform()/2
    img[:, :, 2] = img[:, :, 2] * random_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def main():
    classes = [0, 1, 2, 3, 4, 5]
    nb_images_to_generate_per_class = 100
    final_size = (200, 200)

    orig_path = '../data/originals/'
    dest_path = '../data/augmented/'
    available_transformations = {'rotate': rotate,
                                 'shift x and y': shift,
                                 'zoom in and crop': zoomin,
                                 'add noise': noise,
                                 'blur': blur,
                                 'shear': shear,
                                 'brightness': brightness}

    if len(os.listdir(dest_path)) > 0:
        maxid = max([int(re.findall("[0-9]+", i)[2]) for i in os.listdir(dest_path)])
    else:
        maxid = 1
    images_list = os.listdir(orig_path)

    for c in classes:
        images_list_class = [f for f in images_list if int(f.split('_')[0]) == c]
        for nb in range(0, nb_images_to_generate_per_class):

            # Select a random image
            image_file = random.choice(images_list_class)
            fname, ext = os.path.splitext(image_file)
            image = cv2.imread(os.path.join(orig_path, image_file))

            # Define nb of transformation to apply
            nb_transform = random.randint(1, len(available_transformations))

            # Apply random transform
            for t in range(0, nb_transform):
                t_select = random.choice(list(available_transformations))
                img = available_transformations[t_select](image)

            # Resize img
            img = cv2.resize(img, final_size)

            # Save new image
            new_file_name = "from_%s_%d%s" % (fname, maxid, ext)
            cv2.imwrite(os.path.join(dest_path, new_file_name), img)
            maxid += 1


if __name__ == "__main__":
    main()
