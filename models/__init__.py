from abc import ABC, abstractmethod
from methylnet_utils import split_methylation_array_by_pheno
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import pandas as pd
import pickle


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
            epochs=epochs,
            callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)],
            validation_data=validation_set
        )
        self.__model.save_weights(os.path.join(self.__serialization_path, self.__model_name + ".h5"))


class MethylationArrayGenerator(Sequence):
    """
    This class creates a generator for tensorflow starting from a methylation array
    """

    def __init__(self, df_source, label_column, batch_size=32, shuffle=True):
        """
        Class constructor
        :param df_source: the .pkl containing the pandas dataframe or the dataframe itself
        :param batch_size: the batch size
        :param shuffle: True if shuffles the samples in a batch
        """
        if type(df_source) == str:
            self.df = pickle.load(open(df_source, "rb"))
        else:
            self.df = df_source
        self.df["pheno"] = pd.get_dummies(self.df["pheno"][label_column])
        self.batch_size = batch_size
        self.barcodes = list(self.df["pheno"].index.values)
        self.indexes = np.arange(len(self.barcodes))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Calculates the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.barcodes) / self.batch_size))

    def __getitem__(self, index):
        tmp_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        step_barcodes = [self.barcodes[k] for k in tmp_indexes]
        return self.__data_generation(step_barcodes)

    def on_epoch_end(self):
        """
        Updates data indexes after each epoch
        """
        self.indexes = np.arange(len(self.barcodes))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, step_barcodes):
        """
        Generates a batch of data for the step
        :param step_barcodes: the indexes of the rows
        :return: Two numpy array containing features and labels
        """
        return self.df["beta"].loc[step_barcodes].to_numpy(), self.df["pheno"].loc[step_barcodes].to_numpy()


def methylation_array_kcv(dataset_filename, model_class, model_params, output_target, k=10):
    """
    KCV evaluation of a model that implements AbstractClassifier
    :param dataset_filename: the methylation array filename
    :param model_class: the model class
    :param model_params: a dictionary containing the parameters to init the class
    :param output_target: the label
    :param k: the folds
    :return: the average accuracy
    """
    accuracies = []
    for i in range(k):
        training_set, test_set, validation_set = split_methylation_array_by_pheno(dataset_filename, output_target)
        model = model_class(**model_params)
        model.fit(MethylationArrayGenerator(training_set, output_target),
                  MethylationArrayGenerator(validation_set, output_target),
                  500)
        acc = model.evaluate(test_set["beta"].to_numpy(),
                             pd.get_dummies(test_set["pheno"][output_target]).to_numpy())
        accuracies.append(acc)
    return sum(accuracies)/len(accuracies)
