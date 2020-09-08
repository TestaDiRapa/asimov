import pandas as pd
import os


def filter_cpg_islands(folder):
    """
    The function takes a folder containing TCGA methylation files from Illumina 450 and returns a tuple containing the
    number of files elapsed and a dictionary a dictionary where each key is a CpG island and each value is the number
    of files where the island is not NA.
    :param folder: a string containing the folder
    :return: a tuple
    """
    islands = dict()
    files_counter = 0
    for dir_, _, files in os.walk(folder):
        for file in files:
            if file[-3:] == ".txt":
                files_counter += 1
                dataset = pd.read_csv(os.path.join(dir_, file), sep='\t', na_values="NA")
                print(dataset.size)
                dataset = dataset[["Composite Element REF", "Beta_value"]].dropna()
                print(dataset.size)
                for cpg in dataset.iloc[:, 0]:
                    if cpg in islands:
                        islands[cpg] += 1
                    else:
                        islands[cpg] = 1
    print("Files elapsed: {}".format(files_counter))
    return islands, files_counter
