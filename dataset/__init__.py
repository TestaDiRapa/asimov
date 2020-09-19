from bs4 import BeautifulSoup
import os
import pandas as pd
import re

# Just a dict to avoid using 4 strings for 3 values
STATUS_CODE = {
    "Negative": '-',
    "Positive": '+',
    "Equivocal": '?',
    "Indeterminate": '?'
}


def find_receptor_status(soup, selector):
    """
    The function takes as input the BeautifulSoup-parsed XML tree and a selector and returns the value of the tag if
    the tag exists and the procurement was actually completed.
    :param soup: a BeautifulSoup object
    :param selector: the tag name as string
    :return: the tag value or ?
    """
    tag = soup.find(name=selector)
    if tag is None or tag["procurement_status"] != "Completed":
        return '?'
    return STATUS_CODE[tag.text]


def identify_breast_cancer_subtype(clinical_filename):
    """
    The function takes as input the file name of a clinical file from TCGA and returns the breast cancer subtype
    according to the er, pr and her2 receptor statuses.
    :param clinical_filename: the filename of the clinical file from TCGA
    :return: a string containing the subtype of breast cancer
    """
    with open(clinical_filename, 'r', encoding="utf-8") as input_file:
        raw_xml = input_file.read()

    soup = BeautifulSoup(raw_xml, 'lxml')

    her2 = find_receptor_status(soup, "brca_shared:lab_procedure_her2_neu_in_situ_hybrid_outcome_type")
    if her2 == '?':
        her2 = find_receptor_status(soup, "brca_shared:lab_proc_her2_neu_immunohistochemistry_receptor_status")

    er = find_receptor_status(soup, "brca_shared:breast_carcinoma_estrogen_receptor_status")

    pr = find_receptor_status(soup, "brca_shared:metastatic_breast_carcinoma_progesterone_receptor_status")

    if (er == '+' or pr == '+') and her2 == '-':
        return "LuminalA"

    elif (er == '+' or pr == '+') and her2 == '+':
        return "LuminalB"

    elif (er == '-' or pr == '-') and her2 == '-':
        return "TNBS"

    elif (er == '-' or pr == '-') and her2 == '+':
        return "HER2+"

    else:
        return "Unclear"


def folder_generator(folders, file_filter):
    """
    Generator that yields the path of all the files that match a certain regex in a list of subfolders
    :param folders: An iterable containing the folders or a folder name
    :param file_filter: a regex
    :return: a file path
    """
    if type(folders) is not list:
        folders = [folders]
    for folder in folders:
        for dir_, _, files in os.walk(folder):
            for file in files:
                if re.search(file_filter, file):
                    yield os.path.join(dir_, file)


def feature_counter(folders, folders_filter, index_column, columns_filter=None):
    """
    The function takes a folder containing TCGA methylation files from Illumina 450 and returns a tuple containing the
    number of files elapsed and a dictionary a dictionary where each key is a CpG island and each value is the number
    of files where the island is not NA.
    :param folders: a list of string containing the folders
    :param folders_filter: the regex fir the folder_generator
    :param index_column: the column containing the features
    :param columns_filter: the columns to include to exclude NA values
    :return: a tuple
    """
    islands = dict()
    files_counter = 0
    for path in folder_generator(folders, folders_filter):
        files_counter += 1
        dataset = pd.read_csv(path, sep='\t', na_values="NA")
        if columns_filter is not None:
            dataset = dataset[[index_column]+columns_filter]
        dataset = dataset.dropna()
        for cpg in dataset.loc[:, index_column]:
            if cpg in islands:
                islands[cpg] += 1
            else:
                islands[cpg] = 1
    return islands, files_counter


def filter_expression_by_rate(dataset, rate):
    """
    This function takes a dataset MethylationArray-beta-like and returns all the columns that have a rate of non-zero
    values higher than the rate parameter
    :param dataset: the dataset
    :param rate: the threshold for non-zero values
    :return: a list of columns
    """
    total_rows = dataset.shape[0]
    final_columns = []
    for column in dataset.columns:
        non_zero_rows = dataset[dataset[column] > 0].shape[0]
        if non_zero_rows/total_rows > rate:
            final_columns.append(column)
    return final_columns
