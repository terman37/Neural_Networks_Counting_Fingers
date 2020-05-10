import os
import random
import cv2
import numpy as np
import shutil

def rotate(img):
    h, w = img.shape
    bounds = [-30, 30]  # bounds for the rotation angle in degrees
    r = random.randint(bounds[0], bounds[1])
    rmatr = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=r, scale=1)
    img = cv2.warpAffine(img, rmatr, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def shift(img):
    h, w = img.shape
    bounds = [-25, 25]  # bounds for the shift in x and y in pixels
    sx = random.randint(bounds[0], bounds[1])
    sy = random.randint(bounds[0], bounds[1])
    smatr = np.float32([[1, 0, sx], [0, 1, sy]])
    img = cv2.warpAffine(img, smatr, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def zoomin(img):
    h, w = img.shape
    bounds = [0, 25]  # bounds for the zoom factor in %
    zx = 1 + random.randint(bounds[0], bounds[1]) / 100
    zy = 1 + random.randint(bounds[0], bounds[1]) / 100
    inc_w = round((w * zx) - w)
    inc_h = round((h * zy) - h)
    img = cv2.resize(img, (w + inc_w, h + inc_h), interpolation=cv2.INTER_AREA)
    img = img[inc_h // 2:inc_h // 2 + h, inc_w // 2:inc_w // 2 + w]
    return img


def noise(img):
    h, w = img.shape
    bounds = [0.1, 0.75]
    s2 = random.uniform(bounds[0], bounds[1])
    gauss = np.random.normal(0, s2, img.shape).astype('uint8')
    # Add the Gaussian noise to the image
    img = cv2.add(img, gauss)
    return img


def blur(img):
    bounds = [1, 5]  # bounds for the kernel size
    k = random.randint(bounds[0], bounds[1]) * 2 + 1
    img = cv2.medianBlur(img, k, cv2.BORDER_REPLICATE)
    return img


def shear(img):
    h, w = img.shape
    shear_range = 25
    pts1 = np.float32([[5, 5], [100, 5], [5, 100]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 100 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, shear_M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def augment_list(im_list, nb_per_class, in_folder, out_folder, f_size, max_id, cl):

    available_transformations = {'rotate': rotate,
                                 'shift x and y': shift,
                                 'zoom in and crop': zoomin,
                                 'add noise': noise,
                                 'blur': blur,
                                 'shear': shear,
                                 }

    for nb in range(0, nb_per_class):

        # Select a random image
        image_file = random.choice(im_list)
        fname, ext = os.path.splitext(image_file)
        image = cv2.imread(os.path.join(in_folder, image_file), cv2.IMREAD_GRAYSCALE)

        # Define nb of transformation to apply
        nb_transform = random.randint(1, len(available_transformations))

        # Apply random transform
        for t in range(0, nb_transform):
            t_select = random.choice(list(available_transformations))
            img = available_transformations[t_select](image)

        # Resize img
        img = cv2.resize(img, f_size)

        # Save new image
        if not os.path.exists(os.path.join(out_folder, str(cl))):
            os.mkdir(os.path.join(out_folder, str(cl)))
        new_file_name = "from_%s_%d%s" % (fname, max_id, ext)
        cv2.imwrite(os.path.join(out_folder, str(cl), new_file_name), img)
        max_id += 1

    return max_id


def main():
    classes = [0, 1, 2, 3, 4, 5]
    nb_images_to_generate_per_class = 1000
    final_size = (100, 100)

    orig_path = '../data/originals/'
    dest_path = '../data/train/'
    dest_test_path = '../data/test/'
    dest_val_path = '../data/val/'

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    if os.path.exists(dest_test_path):
        shutil.rmtree(dest_test_path)
    if os.path.exists(dest_val_path):
        shutil.rmtree(dest_val_path)

    os.mkdir(dest_path)
    os.mkdir(dest_test_path)
    os.mkdir(dest_val_path)

    maxid = 1

    images_list = os.listdir(orig_path)

    for c in classes:
        full_images_list_class = [f for f in images_list if int(f.split('_')[0]) == c]

        # keep 20% of originals as images for test (never seen before)
        ntest = round(len(full_images_list_class) * 0.20)
        random.shuffle(full_images_list_class)

        # augment test dataset
        maxid = augment_list(full_images_list_class[:ntest],
                             nb_per_class=int(nb_images_to_generate_per_class * 0.2),
                             in_folder=orig_path,
                             out_folder=dest_test_path,
                             f_size=final_size,
                             max_id=maxid,
                             cl=c)

        # keep 15% of originals as images for validation
        nval = round(len(full_images_list_class[ntest:]) * 0.15)

        # augment val dataset
        maxid = augment_list(full_images_list_class[ntest:ntest + nval],
                             nb_per_class=int(nb_images_to_generate_per_class * 0.15),
                             in_folder=orig_path,
                             out_folder=dest_val_path,
                             f_size=final_size,
                             max_id=maxid,
                             cl=c)

        # augment train dataset
        maxid = augment_list(full_images_list_class[ntest + nval:],
                             nb_per_class=nb_images_to_generate_per_class,
                             in_folder=orig_path,
                             out_folder=dest_path,
                             f_size=final_size,
                             max_id=maxid,
                             cl=c)


if __name__ == "__main__":
    main()
