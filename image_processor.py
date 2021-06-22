from os import listdir

import cv2
import numpy
from tensorflow.python.keras.preprocessing.image import img_to_array

import constant_values


def convert_image_to_array(image_dir):
    print("[INFO] Converting image to array")
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, constant_values.DEFAULT_IMAGE_SIZE)
            print("[INFO] Convert image to array done")
            return img_to_array(image)
        else:
            return numpy.array([])
    except Exception as error:
        print(f"Error{error}")
        return None


def load_images():
    print("[INFO] Loading training Image Dataset ")
    image_list, labels_list = [], []
    try:
        melanoma_disease_folder = listdir(constant_values.TRAIN_DIR)
        for melanoma_folder in melanoma_disease_folder:
            melanoma_image_list = listdir(f"{constant_values.TRAIN_DIR}/{melanoma_folder}/")
            for image in melanoma_image_list[:constant_values.N_IMAGES]:
                single_image = f"{constant_values.TRAIN_DIR}/{melanoma_folder}/{image}"
                if single_image.endswith(".jpg") == True or single_image.endswith(
                        ".JPG") == True or single_image.endswith(
                        ".JPEG") == True:
                    image_list.append(convert_image_to_array(single_image))
                    labels_list.append(melanoma_folder)
        print("[INFO] Image loading complete")
        return image_list, labels_list
    except Exception as e:
        print(repr(e))
