from dataset import identify_breast_cancer_subtype
from dataset import folder_generator
import pandas as pd
import pickle
import random
import time


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
    no_match = []
    for barcode in barcodes:
        if barcode.split('-')[3][:2] == "11":
            pheno.loc[barcode] = "Control"
        else:
            subject_barcode = '-'.join(barcode.split('-')[:3])
            if subject_barcode in subtypes:
                pheno.loc[barcode] = subtypes[subject_barcode]
            else:
                no_match.append(barcode)
    clean_df = dataset_w_barcodes.drop(no_match)
    pheno = pheno.drop(no_match)
    return {"beta": clean_df, "pheno": pheno}


def split_methylation_array_by_pheno(methylation_array_source, pheno_column, val_rate=0.1, test_rate=0.1, seed=None):
    """
    This function take a methylation array and split it into training, validation and test set keeping the classes
    balancing
    :param methylation_array_source: the methylation array filename or the methylation array itself
    :param pheno_column: the column containing the classes
    :param val_rate: the rate of samples to include in the validation set
    :param test_rate: the rate of samples to include in the test set
    :param seed: the random seed for the split
    :return train, test and validation set
    """
    if seed is None:
        random.seed(int(time.time()*100000000))
    else:
        random.seed(seed)
    if type(methylation_array_source) == str:
        methylation_array = pickle.load(open(methylation_array_source, "rb"))
    else:
        methylation_array = methylation_array_source
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


def cut_index_barcode(methylation_array):
    new_pheno = methylation_array["pheno"]
    new_pheno["barcode"] = ""
    new_beta = methylation_array["beta"]
    new_beta["barcode"] = ""

    for barcode in new_pheno.index.values:
        new_barcode = "-".join(barcode.split("-")[:4])
        new_pheno.at[barcode, "barcode"] = new_barcode
        new_beta.at[barcode, "barcode"] = new_barcode

    new_pheno = new_pheno.set_index("barcode")
    new_beta = new_beta.set_index("barcode")
    return {"beta": new_beta, "pheno": new_pheno}


def merge_methylation_arrays(*args):
    """
    This functions merges several methylation arrays into a single methylation array using the common row by index. The
    barcode-subtype couple in each pheno must be the same
    :param args: the methylation arrays
    :return: a single methylation array
    """
    barcodes_all = []
    reindexed_arrays = []
    for methylation_array in args:
        # reindexed_ma = cut_index_barcode(methylation_array)
        reindexed_ma = {
            "pheno": methylation_array["pheno"].rename(index=lambda barcode: "-".join(barcode.split("-")[:4])),
            "beta": methylation_array["beta"].rename(index=lambda barcode: "-".join(barcode.split("-")[:4]))
        }
        barcodes_all.append(set(reindexed_ma["pheno"].index.drop_duplicates(keep=False).values))
        reindexed_arrays.append(reindexed_ma)

    barcodes = barcodes_all[0]
    for b in barcodes_all[1:]:
        barcodes = barcodes.intersection(b)

    final_beta = pd.concat([methylation_array["beta"].loc[barcodes] for methylation_array in reindexed_arrays], axis=1)
    return {"beta": final_beta, "pheno": reindexed_arrays[0]["pheno"].loc[barcodes]}
