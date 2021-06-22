from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

import constant_values
from file_utils import save_model


def create_model(n_classes):
    print("[INFO] Creating model...")
    model = Sequential()
    input_shape = (constant_values.IMG_HEIGHT, constant_values.IMG_WIDTH, constant_values.DEPTH)
    channel_dimension = -1
    if backend.image_data_format() == "channels_first":
        input_shape = (constant_values.DEPTH, constant_values.IMG_HEIGHT, constant_values.IMG_WIDTH)
        channel_dimension = 1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dimension))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dimension))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dimension))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    model.summary()
    print("[INFO] Creating model Complete")
    return model


def compile_train_and_save_model(model, augment, x_train, y_train, x_test, y_test):
    print("[INFO] Compiling and Training the model")
    optim = Adam(lr=constant_values.LR, decay=constant_values.LR / constant_values.TRAIN_EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=optim, metrics="accuracy")
    print("[INFO] Compiling complete")
    print("[INFO] Training...")

    history = model.fit(augment.flow(x_train, y_train, batch_size=constant_values.BATCH_SIZE),
                        validation_data=(x_test, y_test),
                        steps_per_epoch=len(x_train) // constant_values.BATCH_SIZE,
                        epochs=constant_values.TRAIN_EPOCHS,
                        verbose=1)
    print("[INFO] Training complete")
    save_model(model)
    return model


def get_model_accuracy(model, x_test, y_test):
    print("[INFO] Getting model accuracy")
    accuracy = model.evaluate(x_test, y_test)
    print("[INFO] Done")
    return accuracy[1] * 100
