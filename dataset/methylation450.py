import pandas as pd
import os
import re


def create_methylation_dataset(folders, islands):
    """
    The function creates a pandas Dataframe where each row is a sample and each column is a CpG island. Only the islands
    passed as parameter are considered.
    :param folders: a list of string containing the folders where the files are
    :param islands: a list of CpG island
    :return: a pandas Dataframe
    """
    new_dataset = pd.DataFrame(columns=islands+["barcode"])
    count = 1
    for path in folder_generator(folders, r'^jhu-usc\..+txt$'):
        print(count)
        count += 1
        dataset = pd.read_csv(path, sep='\t', na_values="NA", index_col=0)
        dataset = dataset[["Beta_value"]].dropna().loc[islands].T.reset_index().drop("index", axis=1)
        dataset.columns.name = None
        barcode = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]{3}-[A-Z0-9]{4}-[A-Z0-9]{2}', path).group()
        dataset["barcode"] = barcode
        new_dataset = new_dataset.append(dataset)
    new_dataset.set_index("barcode")
    return new_dataset


def filter_cpg_islands(folders):
    """
    The function takes a folder containing TCGA methylation files from Illumina 450 and returns a tuple containing the
    number of files elapsed and a dictionary a dictionary where each key is a CpG island and each value is the number
    of files where the island is not NA.
    :param folders: a list of string containing the folders
    :return: a tuple
    """
    islands = dict()
    files_counter = 0
    for path in folder_generator(folders, r'^jhu-usc\..+txt$'):
        files_counter += 1
        dataset = pd.read_csv(path, sep='\t', na_values="NA")
        dataset = dataset[["Composite Element REF", "Beta_value"]].dropna()
        for cpg in dataset.iloc[:, 0]:
            if cpg in islands:
                islands[cpg] += 1
            else:
                islands[cpg] = 1
    return islands, files_counter


def folder_generator(folders, file_filter):
    """
    Generator that yields the path of all the files that match a certain regex in a list of subfolders
    :param folders: An iterable containing the folders
    :param file_filter: a regex
    :return: a file path
    """
    for folder in folders:
        for dir_, _, files in os.walk(folder):
            for file in files:
                if re.search(file_filter, file):
                    yield os.path.join(dir_, file)
