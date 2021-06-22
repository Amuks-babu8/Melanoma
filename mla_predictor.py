import numpy

from image_processor import convert_image_to_array


def predict_melanoma(model, path, l_binary):
    image_in_array_form = convert_image_to_array(path)
    result = get_prediction_result(model, image_in_array_form)
    image_class = l_binary.classes_[result]
    prediction = image_class[0]
    return prediction


def get_prediction_result(model, arr_img):
    numpy_image = numpy.array(arr_img, dtype=numpy.float16)/225.0
    numpy_image = numpy.expand_dims(numpy_image, 0)
    result = numpy.argmax(model.predict(numpy_image), axis=-1)
    return result
