from dataset import BarcodeFinder, folder_generator
import os
import pandas as pd


def mrna_dataset_creator(mrna_folders, genes, barcode=True):
    """
    This function creates a dictionary where the keys are the mRBA filenames and the values the barcodes of the
    subjects
    :param mrna_folders: the folder or a list of folders that contains all the mRNA files
    :param genes: a list of transcripts to consider
    :param barcode: boolean flag to search the barcode on GDC or use the uuid instead
    :return: a dictionary
    """
    finder = BarcodeFinder()
    new_dataset = pd.DataFrame(columns=genes + ["barcode"])
    for path in folder_generator(mrna_folders, r'FPKM\.txt\.gz$'):
        dataset = pd.read_csv(path, compression="gzip", sep='\t', index_col=0, header=None, na_values="NA")
        dataset = dataset.dropna().loc[genes].T
        dataset.columns.name = None
        if barcode:
            dataset["barcode"] = finder.find_barcode(path.split(os.sep)[-2])
        else:
            dataset["barcode"] = path.split(os.sep)[-2]
        new_dataset = new_dataset.append(dataset)
    finder.quit()
    new_dataset = new_dataset.set_index("barcode")
    return new_dataset
