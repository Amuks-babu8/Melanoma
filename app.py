import math
import os

import numpy
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import constant_values
from file_utils import save_labels, load_labels, load_saved_model
from image_processor import load_images
from mla_predictor import predict_melanoma
from mla_training_trainer import create_model, compile_train_and_save_model, get_model_accuracy

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = "Images"


@app.route('/')
def hello_world():
    return render_template('index.html')


labels_bin_array = None


@app.route('/check', methods=['POST'])
def check_melanoma():
    global labels_bin_array
    image_to_check = request.files['skin']
    path = os.path.join(app.config['UPLOAD_FOLDER'], image_to_check.filename)
    image_to_check.save(path)

    # load dataset
    image_list, labels_list = load_images()

    # convert the loaded training dataset into numpy array
    np_image_list = numpy.array(image_list, dtype=numpy.float16) / 225.0

    labels_bin_array = LabelBinarizer()
    image_labels = labels_bin_array.fit_transform(labels_list)

    n_classes = len(labels_bin_array.classes_)

    # save labels to pickle file
    save_labels(labels_bin_array)

    # split the dataset to training print("[INFO] Splitting the data to train and test")
    xtrain, xtest, ytrain, ytest = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

    # check if model is already present and just use it instead of compiling a new one
    if os.path.exists(constant_values.LABELS_FILE) and os.path.exists(constant_values.MODEL_FILE):
        labels_bin_array = load_labels()
        model = load_saved_model()
        accuracy = get_model_accuracy(model, xtest, ytest)
        accuracy = math.ceil(accuracy)
        prediction = predict_melanoma(model, path, labels_bin_array)
        return render_template("index.html", status=prediction, accuracy=accuracy)

    # perform augmentation to increase accuracy
    augment = ImageDataGenerator(rotation_range=25,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode="nearest")

    model = create_model(1)
    model = compile_train_and_save_model(model, augment, xtrain, ytrain, xtest, ytest)

    accuracy = get_model_accuracy(model, xtest, ytest)
    accuracy = math.ceil(accuracy)

    prediction = predict_melanoma(model, path, labels_bin_array)

    return render_template("index.html", status=prediction, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
