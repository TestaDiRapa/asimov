import pandas as pd
import os
import re


def folder_generator(folders, file_filter):
    for folder in folders:
        for dir_, _, files in os.walk(folder):
            for file in files:
                if re.search(file_filter, file):
                    yield os.path.join(dir_, file)


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
