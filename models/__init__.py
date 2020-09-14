from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import pickle


class MethylationArrayGenerator(Sequence):

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
    dataset = pickle.load(open(dataset_filename, "rb"))
    barcodes = list(dataset["pheno"].index.values)
    partition_size = int(len(barcodes)/k)
    barcodes_partitions = []
    for i in range(k):
        barcodes_partitions.append(barcodes[i*partition_size:(i+1)*partition_size])

    accuracies = []
    for val_i, test_i in zip(range(-1, k-1), range(k)):
        tmp_partitions = [b for b in barcodes_partitions]
        val_barcodes = tmp_partitions.pop(val_i)
        test_barcodes = tmp_partitions.pop(test_i)
        train_barcodes = []
        for b in tmp_partitions:
            train_barcodes += b

        training_set = MethylationArrayGenerator(
            {"pheno": dataset["pheno"].loc[train_barcodes], "beta": dataset["beta"].loc[train_barcodes]},
            output_target
        )
        validation_set = MethylationArrayGenerator(
            {"pheno": dataset["pheno"].loc[val_barcodes], "beta": dataset["beta"].loc[val_barcodes]},
            output_target
        )

        model = model_class(**model_params)
        model.fit(training_set, validation_set, 500)
        acc = model.predict(dataset["beta"].loc[test_barcodes].to_numpy(),
                            pd.get_dummies(dataset["pheno"][output_target]).to_numpy())
        accuracies.append(acc)
    return sum(accuracies)/len(accuracies)
