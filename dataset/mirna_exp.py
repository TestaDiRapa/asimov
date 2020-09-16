from dataset import folder_generator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import pandas as pd


def mirna_dataset_creator(mirna_folder, mirnas):
    """
    This function creates a dictionary where the keys are the mirna filenames and the values the barcodes of the
    subjects
    :param mirna_folder: the folder that contains all the mirna files
    :param mirnas: a list of mirna to consider
    :return: a dictionary
    """
    driver = webdriver.Firefox()
    driver.maximize_window()
    new_dataset = pd.DataFrame(columns=mirnas + ["barcode"])
    for folder in folder_generator(mirna_folder, r'mirnas\.quantification\.txt$'):
        dataset = pd.read_csv(folder, sep='\t', na_values="NA", index_col=0)
        dataset = dataset[["reads_per_million_miRNA_mapped"]].dropna().loc[mirnas].T.reset_index().drop("index", axis=1)
        dataset.columns.name = None
        url = "https://portal.gdc.cancer.gov/files/{}".format(folder.split(os.sep)[-2])
        driver.get(url)
        css_selector = "a.unnamed-link[href*='bioId']"
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))
        barcode = driver.find_element_by_css_selector(css_selector).text
        print(barcode)
        dataset["barcode"] = barcode
        new_dataset = new_dataset.append(dataset)
    new_dataset.set_index("barcode")
    return new_dataset
