from abc import ABC, abstractmethod
from methylnet_utils import split_methylation_array_by_pheno
from models.generators import MethylationArrayGenerator
from tensorflow.keras.models import Model
import numpy as np
import os
import pandas as pd


class AbstractModel(ABC):
    """
    An abstract class to define a common interface for all the models
    """

    def __init__(self,  model_serialization_path="models/autoencoder/", model_name="autoencoder"):
        """
        Class constructor
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__()
        self.__model = None
        if not os.path.exists(model_serialization_path):
            os.makedirs(model_serialization_path)
        self._serialization_path = model_serialization_path
        self._model_name = model_name

    @abstractmethod
    def generate_model(self, input_shape, output_shape):
        """
        Method that the subclasses must implement, defines the model itself
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        pass

    def save_model(self):
        self.__model.save(os.path.join(self._serialization_path, self._model_name + ".h5"))

    def save_weights(self):
        self.__model.save_weights(os.path.join(self._serialization_path, self._model_name + ".h5"))

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
        self.__model.summary()

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

    def fit(self, training_set, validation_set, epochs, callbacks=[], verbose=1):
        """
        This method fits the model using the EarlyStopping callback and save the weights
        :param training_set: the training set
        :param validation_set:  the validation set
        :param epochs: the number of epochs
        :param callbacks: the callbacks to pass to fit method
        :param verbose: 0, 1 or 2 (check tensorflow documentation)
        :return: None
        """
        self.__model.fit(
            training_set,
            verbose=verbose,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_set
        )


def methylation_array_kcv(dataset, model_class, model_params, output_target, k=10, verbose=0, callbacks=[]):
    """
    KCV evaluation of a model that implements AbstractClassifier
    :param dataset: the methylation array filename or the methylation array itself
    :param model_class: the model class
    :param model_params: a dictionary containing the parameters to init the class
    :param output_target: the label
    :param k: the folds
    :param verbose: verbose mode for fit method
    :param callbacks: the callbacks to pass to the fit method
    :return: the average accuracy
    """
    test_accuracies, val_accuracies = [], []
    for i in range(k):
        training_set, test_set, validation_set = split_methylation_array_by_pheno(dataset, output_target)
        model = model_class(**model_params)
        model.fit(MethylationArrayGenerator(training_set, output_target),
                  MethylationArrayGenerator(validation_set, output_target),
                  500,
                  verbose=verbose,
                  callbacks=callbacks)
        test_accuracies.append(model.evaluate(test_set["beta"].to_numpy(),
                               pd.get_dummies(test_set["pheno"][output_target]).to_numpy()))
        val_accuracies.append(model.evaluate(validation_set["beta"].to_numpy(),
                                             validation_set["pheno"].to_numpy()))
    return sum(val_accuracies)/len(val_accuracies), sum(test_accuracies)/len(test_accuracies)
