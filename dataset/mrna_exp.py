from dataset import BarcodeFinder, folder_generator
import os
import pandas as pd


def mrna_dataset_creator(mrna_folder, genes):
    """
    This function creates a dictionary where the keys are the mRBA filenames and the values the barcodes of the
    subjects
    :param mrna_folder: the folder that contains all the mRNA files
    :param genes: a list of transcripts to consider
    :return: a dictionary
    """
    finder = BarcodeFinder()
    new_dataset = pd.DataFrame(columns=genes + ["barcode"])
    for folder in folder_generator(mrna_folder, r'FPKM\.txt\.gz$'):
        dataset = pd.read_csv(folder, compression="gzip", sep='\t', index_col=0, header=None, na_values="NA")
        dataset = dataset.dropna().loc[genes]
        dataset.columns.name = None
        dataset["barcode"] = finder.find_barcode(folder.split(os.sep)[-2])
        new_dataset = new_dataset.append(dataset)
    finder.quit()
    new_dataset = new_dataset.set_index("barcode")
    return new_dataset
