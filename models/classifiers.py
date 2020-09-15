from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


class AbstractClassifier(ABC):
    """
    An abstract class that define a common fit/prediction interface for all the models
    """

    def __init__(self,  model_serialization_path="models/classifier/", model_name="classifier"):
        """
        Class constructor
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__()
        self.__model = None
        if not os.path.exists(model_serialization_path):
            os.makedirs(model_serialization_path)
        self.__serialization_path = model_serialization_path
        self.__model_name = model_name

    @abstractmethod
    def generate_model(self, input_shape):
        """
        Method that the subclasses must implement, defines the model itself
        :param input_shape: the input shape
        :return: None
        """
        pass

    def compile_model(self, input_layer, output_layer, optimizer, loss):
        """
        This method compiles the method and saves it
        :param input_layer: the input layer (a keras layer)
        :param output_layer: the output layer (a keras layer)
        :param optimizer: the optimizer function
        :param loss: the loss function
        :return: None
        """
        self.__model = Model(inputs=input_layer, outputs=output_layer)

        self.__model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=["accuracy"])
        self.__model.save(os.path.join(self.__serialization_path, self.__model_name + ".h5"))
        self.__model.summary()

    def fit(self, training_set, validation_set, epochs, verbose=1):
        """
        This method fits the model using the EarlyStopping callback and save the weights
        :param training_set: the training set
        :param validation_set:  the validation set
        :param epochs: the number of epochs
        :param verbose: 0, 1 or 2 (check tensorflow documentation)
        :return: None
        """
        self.__model.fit(
            training_set,
            verbose=verbose,
            # batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)],
            validation_data=validation_set
        )
        self.__model.save_weights(os.path.join(self.__serialization_path, self.__model_name+".h5"))

    def predict(self, x_test):
        """
        Wrapper for the predict method of the model class from keras
        :param x_test: the test set
        :return: the predictions
        """
        return self.__model.predict(x_test)

    def evaluate(self, x_test, y_test):
        """
        This method evaluates the performances of the model on a test set
        :param x_test: the test set features
        :param y_test: the test set labels
        :return: the average accuracy
        """
        acc = 0
        for p, r in zip(self.predict(x_test), y_test):
            if np.argmax(p) == np.argmax(r):
                acc += 1
        return acc/len(y_test)


class NeuralClassifier(AbstractClassifier):
    """
    Implements the abstract model as a fully connected NN with 3 hidden layers
    """

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape)

    def generate_model(self, input_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        hl_1 = Dense(200, activation="relu")(input_layer)
        hl_2 = Dense(300, activation="relu")(hl_1)
        hl_3 = Dense(3000, activation="relu")(hl_2)
        hl_d = Dropout(self.dropout_rate)(hl_3)
        output_layer = Dense(6, activation="sigmoid")(hl_d)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())


class ConvolutionalClassifier(AbstractClassifier):
    """
    Implements the abstract model as a convolutional neural network
    """

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape)

    def generate_model(self, input_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        reshaped_input = Reshape((100, 1))(input_layer)
        conv_1 = Conv1D(128, 4, activation="relu")(reshaped_input)
        conv_2 = Conv1D(64, 2, activation="relu")(conv_1)
        drop_l = Dropout(self.dropout_rate)(conv_2)
        conv_3 = Conv1D(64, 2, activation="relu")(drop_l)
        flat = Flatten()(conv_3)
        drop_2 = Dropout(self.dropout_rate)(flat)
        output_layer = Dense(6, activation="sigmoid")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())
