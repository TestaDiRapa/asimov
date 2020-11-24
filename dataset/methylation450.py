from dataset import folder_generator
import pandas as pd
import re


def create_methylation_dataset(folders, islands=None, filters=dict()):
    """
    The function creates a pandas Dataframe where each row is a sample and each column is a CpG island. Only the islands
    passed as parameter are considered.
    :param folders: a list of string containing the folders where the files are
    :param islands: a list of CpG island
    :param filters: a dict where the key is the column and the value the value used to filter
    :return: a pandas Dataframe
    """
    for path in folder_generator(folders, r'^jhu-usc\..+txt$'):
        dataset = pd.read_csv(path, sep='\t', na_values="NA", index_col=0)
        dataset.dropna()
        if islands is not None:
            dataset = dataset.loc[islands]
        for col, value in filters.items():
            dataset = dataset[dataset[col] == value]
        filtered_islands = list(dataset.index.values)
        break

    new_dataset = pd.DataFrame(columns=filtered_islands+["barcode"])
    count = 0
    for path in folder_generator(folders, r'^jhu-usc\..+txt$'):
        try:
            dataset = pd.read_csv(path, sep='\t', na_values="NA", index_col=0)
            dataset = dataset[["Beta_value"]].loc[filtered_islands].T.reset_index().drop("index", axis=1)
            dataset.columns.name = None
            barcode = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]{3}-[A-Z0-9]{4}-[A-Z0-9]{2}', path).group()
            dataset["barcode"] = barcode
            new_dataset = new_dataset.append(dataset)
            count += 1
            print(count)
        except KeyError:
            print("Skipping file")
    new_dataset = new_dataset.set_index("barcode")
    print(new_dataset)
    new_dataset = new_dataset.dropna(axis=1)
    print()
    print(new_dataset)
    return new_dataset
