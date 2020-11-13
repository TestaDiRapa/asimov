import pickle


class OmicArrayInitError(Exception):
    pass


class OmicArray:
    """
    A class to represent the MethylationArray data type. Renamed to OmicArray for clarity
    """

    def __init__(self, filename=None, beta=None, pheno=None):
        """
        Class constructor.
        :param filename: a .pkl file containing a dict with beta and pheno pandas dataframes
        :param beta: the beta values to init the object directly
        :param pheno: the pheno values to init the object directly
        """
        if filename is None and beta is None and pheno is None:
            raise OmicArrayInitError("An input source must be specified")
        elif filename is not None and (beta is not None or pheno is not None):
            raise OmicArrayInitError("Only one among filename and beta/pheno must be specified")
        elif (beta is None and pheno is not None) or (beta is not None and pheno is None):
            raise OmicArrayInitError("Both pheno and beta must be specified")

        if filename is not None:
            tmp = pickle.load(open(filename, "rb"))
            self.beta = tmp["beta"]
            self.pheno = tmp["pheno"]
        else:
            self.beta = beta
            self.pheno = pheno

    def deep_copy(self):
        """
        Creates a deep copy of the structure
        :return: an OmicArray
        """
        return OmicArray(beta=self.beta.copy(), pheno=self.pheno.copy())

    def pheno_replace(self, condition, column, value, inplace=True):
        """
        Replace values in pheno
        :param condition: a boolean array representing each row in the
        :param column: the column where the values are to be replaced
        :param value: the value to replace
        :param inplace: boolean value. False if a new OmicArray must be created
        :return: an OmicArray if inplace is False
        """
        if inplace:
            self.pheno.loc[condition, column] = value
        else:
            dc = self.deep_copy()
            dc.pheno_replace(condition, column, value)
            return dc

    def sklearn_conversion(self, pheno_column):
        """
        Converts the structure into X and y vectors that can be given as input to sklearn fit method
        :param pheno_column: the pheno column containing the values
        :return: X and y
        """
        return self.beta.to_numpy(), self.pheno[pheno_column].ravel()

    def pheno_unique_values(self, pheno_column):
        """
        Returns a list containing all the unique values contained into a column of the pheno dataframe
        :param pheno_column: the column
        :return: a list
        """
        return list(self.pheno[pheno_column].unique())

    def get_omic_index(self):
        """
        Getter for the index of the beta dataframe
        :return: a list
        """
        return self.beta.index

    def get_pheno_index(self):
        """
        Getter for the index of the pheno dataframe
        :return: a list
        """
        return self.pheno.index

    def get_omic_column_index(self):
        """
        Getter for the column index of the beta dataframe
        :return: a list
        """
        return self.beta.columns

    def get_pheno_column_index(self):
        """
        Getter for the column index of the pheno dataframe
        :return: a list
        """
        return self.pheno.columns

    def append(self, omic_array, check_columns=True):
        """
        Append to the OmicArray another OmicArray with the same keys.
        :param omic_array: Another OmicArray
        :param check_columns: True if the indexes of the columns must be the same
        """
        if check_columns and not self.get_omic_column_index().equals(omic_array.get_omic_column_index()) \
                and not self.get_pheno_column_index().equals(omic_array.get_pheno_column_index()):
            raise Exception("The two omic arrays have different column indexes")

        self.beta.append(omic_array.beta)
        self.pheno.append(omic_array.pheno)

    def __str__(self):
        return "OMIC \n {} values for {} samples \n {} \n PHENO \n {} values for {} samples \n {}".format(
            self.beta.shape[1], self.beta.shape[0], self.beta.head(),
            self.pheno.shape[1], self.pheno.shape[0], self.pheno.head())
