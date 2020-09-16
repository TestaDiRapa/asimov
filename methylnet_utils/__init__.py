from dataset import identify_breast_cancer_subtype
from dataset import folder_generator
import pandas as pd
import pickle
import random


def generate_subtype_methylation_array(clinical_folder, dataset_w_barcodes):
    """
    This function creates a MethylationArray for methylnet using beta values related to samples and clinical data of the
    same samples
    :param clinical_folder: the folder containing the clinical data
    :param dataset_w_barcodes: the values as pandas dataframe with barcodes as index
    """
    subtypes = dict()
    for file in folder_generator(clinical_folder, r'xml$'):
        subject_barcode = file.split('.')[-2]
        subtypes[subject_barcode] = identify_breast_cancer_subtype(file)

    barcodes = list(dataset_w_barcodes.index.values)
    pheno = pd.DataFrame(index=barcodes, columns=["subtype"])
    for barcode in barcodes:
        if barcode.split('-')[3][:2] == "11":
            pheno.loc[barcode] = "Control"
        else:
            subject_barcode = '-'.join(barcode.split('-')[:3])
            pheno.loc[barcode] = subtypes[subject_barcode]
    return {"beta": dataset_w_barcodes, "pheno": pheno}


def split_methylation_array_by_pheno(methylation_array_filename, pheno_column, val_rate=0.1, test_rate=0.1):
    """
    This function take a methylation array and split it into training, validation and test set keeping the classes
    balancing
    :param methylation_array_filename: the methylation array filename
    :param pheno_column: the column containing the classes
    :param val_rate: the rate of samples to include in the validation set
    :param test_rate: the rate of samples to include in the test set
    :return train, test and validation set
    """
    methylation_array = pickle.load(open(methylation_array_filename, "rb"))
    beta = methylation_array["beta"]
    pheno = methylation_array["pheno"]

    train_barcodes, test_barcodes, val_barcodes = [], [], []
    for class_ in pheno[pheno_column].unique():
        barcodes = pheno[pheno[pheno_column] == class_].index.to_list()
        val_samples = int(len(barcodes)*val_rate)
        test_samples = int(len(barcodes)*test_rate)
        samples = random.sample(barcodes, val_samples+test_samples)
        val_tmp = samples[:val_samples]
        test_tmp = samples[val_samples:]
        for b in barcodes:
            if b in val_tmp:
                val_barcodes.append(b)
            elif b in test_tmp:
                test_barcodes.append(b)
            else:
                train_barcodes.append(b)

    return {"beta": beta.loc[train_barcodes], "pheno": pheno.loc[train_barcodes]},\
           {"beta": beta.loc[test_barcodes], "pheno": pheno.loc[test_barcodes]},\
           {"beta": beta.loc[val_barcodes], "pheno": pheno.loc[val_barcodes]}
