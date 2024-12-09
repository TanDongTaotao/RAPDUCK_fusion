import glob

import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import cv2


def load_data(img_height, img_width, images_to_be_loaded, dataset, folder_path, resize=True):
    IMAGES_PATH = folder_path + 'images/'
    MASKS_PATH = folder_path + 'masks/'
    train_ids=[]
    if dataset == 'kvasir':
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    if dataset == 'cvc-clinicdb':
        # train_ids = glob.glob(IMAGES_PATH + "*.tif")
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    if dataset == 'cvc-colondb' or dataset == 'etis-laribpolypdb':
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    if dataset == 'etis':
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    if dataset == 'colon':
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    
    if dataset == 'jpg':
        train_ids = glob.glob(IMAGES_PATH + "*.jpg")
    
    # add ISICDM
    if dataset == 'ISICDM':
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    # 如果不调整大小,则使用第一张图片的尺寸作为数组大小
    if not resize:
        sample_img = cv2.imread(train_ids[0], cv2.IMREAD_COLOR)
        img_height, img_width = sample_img.shape[:2]


    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print('Resizing training images and masks: ' + str(images_to_be_loaded))
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = np.zeros((img_height, img_width), dtype=np.bool_)

        pillow_image = Image.fromarray(image)

        pillow_image = pillow_image.resize((img_height, img_width))
        image = np.array(pillow_image)

        X_train[n] = image / 255

        pillow_mask = Image.fromarray(mask_)
        pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
        mask_ = np.array(pillow_mask)
        
        for i in range(img_height):
            for j in range(img_width):                
                if mask_[i, j] >= 127:                   
                    mask[i, j] = 1

        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1)

    return X_train, Y_train