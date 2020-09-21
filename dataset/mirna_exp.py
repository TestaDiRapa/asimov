from dataset import BarcodeFinder, folder_generator
import os
import pandas as pd


def mirna_dataset_creator(mirna_folder, mirnas, barcode = True):
    """
    This function creates a dictionary where the keys are the mirna filenames and the values the barcodes of the
    subjects
    :param mirna_folder: the folder that contains all the mirna files
    :param mirnas: a list of mirna to consider
    :param barcode: boolean flag to search the barcode on GDC or use the uuid instead
    :return: a dictionary
    """
    new_dataset = pd.DataFrame(columns=mirnas + ["barcode"])
    finder = BarcodeFinder()
    for folder in folder_generator(mirna_folder, r'mirnas\.quantification\.txt$'):
        dataset = pd.read_csv(folder, sep='\t', na_values="NA", index_col=0)
        dataset = dataset[["reads_per_million_miRNA_mapped"]].dropna().loc[mirnas].T.reset_index().drop("index", axis=1)
        dataset.columns.name = None
        if barcode:
            dataset["barcode"] = finder.find_barcode(folder.split(os.sep)[-2])
        else:
            dataset["barcode"] = folder.split(os.sep)[-2]
        new_dataset = new_dataset.append(dataset)
    finder.quit()
    new_dataset = new_dataset.set_index("barcode")
    return new_dataset
