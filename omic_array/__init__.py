import pickle


class OmicArray:
    """
    A class to represent the MethylationArray data type. Renamed to OmicArray for clarity
    """

    def __init__(self, filename):
        """
        Class constructor.
        :param filename: a .pkl file containing a dict with beta and pheno pandas dataframes
        """
        tmp = pickle.load(open(filename, "rb"))
        self.beta = tmp["beta"]
        self.pheno = tmp["pheno"]

    def pheno_replace(self, condition, column, value):
        """
        Replace values in pheno
        :param condition: a boolean array representing each row in the
        :param column: the column where the values are to be replaced
        :param value: the value to replace
        :return:
        """
        self.pheno.loc[condition, column] = value

    def sklearn_conversion(self, pheno_column):
        """
        Converts the structure into X and y vectors that can be given as input to sklearn fit method
        :param pheno_column: the pheno column containing the values
        :return: X and y
        """
        return self.beta.to_numpy(), self.pheno[pheno_column].ravel()

    def __str__(self):
        return "OMIC \n {} values for {} samples \n {} \n PHENO \n {} values for {} samples \n {}".format(
            self.beta.shape[1], self.beta.shape[0], self.beta.head(),
            self.pheno.shape[1], self.pheno.shape[0], self.pheno.head())
