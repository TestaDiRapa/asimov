from abc import ABC
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import pickle


class AbstractGenerator(ABC):
    """
    This class creates a generator for tensorflow starting from a methylation array
    """

    def __init__(self, df_source, barcodes, batch_size, shuffle=True):
        """
        Class constructor
        :param df_source: the .pkl containing the methylation array or the methylation array itself
        :param batch_size: the batch size
        :param barcodes: the barcodes
        :param shuffle: True if shuffles the samples in a batch
        """
        super().__init__()
        if type(df_source) == str:
            self.df = pickle.load(open(df_source, "rb"))
        else:
            self.df = df_source
        self.batch_size = batch_size
        self.barcodes = barcodes
        self.indexes = np.arange(len(self.barcodes))
        self.shuffle = shuffle

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

    def data(self):
        """
        Getter for the data to iterate
        :return: the data
        """
        return self.df


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


class AutoencoderGenerator(Sequence):
    """
    This class creates a generator for tensorflow starting from a methylation array
    """

    def __init__(self, df_source, batch_size=32, shuffle=True):
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
        self.batch_size = batch_size
        self.barcodes = list(self.df.index.values)
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
        return self.df.loc[step_barcodes].to_numpy(), self.df.loc[step_barcodes].to_numpy()
