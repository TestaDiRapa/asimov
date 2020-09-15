from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


class AbstractClassifier(ABC):

    def __init__(self,  model_serialization_path="models/classifier/", model_name="classifier"):
        super().__init__()
        self.__model = None
        if not os.path.exists(model_serialization_path):
            os.makedirs(model_serialization_path)
        self.__serialization_path = model_serialization_path
        self.__model_name = model_name

    @abstractmethod
    def generate_model(self, input_shape):
        pass

    def compile_model(self, input_layer, output_layer, optimizer, loss):
        self.__model = Model(inputs=input_layer, outputs=output_layer)

        self.__model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=["accuracy"])
        self.__model.save(os.path.join(self.__serialization_path, self.__model_name + ".h5"))
        self.__model.summary()

    def fit(self, training_set, validation_set, epochs, verbose=1):
        self.__model.fit(
            training_set,
            verbose=verbose,
            # batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.1, patience=10)],
            validation_data=validation_set
        )
        self.__model.save_weights(os.path.join(self.__serialization_path, self.__model_name+".h5"))

    def predict(self, x_test):
        return self.__model.predict(x_test)

    def evaluate(self, x_test, y_test):
        acc = 0
        for p, r in zip(self.predict(x_test), y_test):
            if np.argmax(p) == np.argmax(r):
                acc += 1
        return acc/len(y_test)


class NeuralClassifier(AbstractClassifier):

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape)

    def generate_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        hl_1 = Dense(200, activation="relu")(input_layer)
        hl_2 = Dense(300, activation="relu")(hl_1)
        hl_3 = Dense(3000, activation="relu")(hl_2)
        hl_d = Dropout(self.dropout_rate)(hl_3)
        output_layer = Dense(6, activation="sigmoid")(hl_d)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())


class ConvolutionalClassifier(AbstractClassifier):

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape)

    def generate_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        hl_1 = Dense(200, activation="relu")(input_layer)
        hl_2 = Dense(300, activation="relu")(hl_1)
        hl_3 = Dense(3000, activation="relu")(hl_2)
        hl_d = Dropout(self.dropout_rate)(hl_3)
        output_layer = Dense(6, activation="sigmoid")(hl_d)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())