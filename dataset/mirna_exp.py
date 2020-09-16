from dataset import folder_generator
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options
import os
import pandas as pd
import time


def mirna_dataset_creator(mirna_folder, mirnas):
    """
    This function creates a dictionary where the keys are the mirna filenames and the values the barcodes of the
    subjects
    :param mirna_folder: the folder that contains all the mirna files
    :param mirnas: a list of mirna to consider
    :return: a dictionary
    """
    driver_options = Options()
    driver_options.headless = True
    driver = webdriver.Firefox(options=driver_options)
    driver.maximize_window()
    new_dataset = pd.DataFrame(columns=mirnas + ["barcode"])
    for folder in folder_generator(mirna_folder, r'mirnas\.quantification\.txt$'):
        dataset = pd.read_csv(folder, sep='\t', na_values="NA", index_col=0)
        dataset = dataset[["reads_per_million_miRNA_mapped"]].dropna().loc[mirnas].T.reset_index().drop("index", axis=1)
        dataset.columns.name = None
        url = "https://portal.gdc.cancer.gov/files/{}".format(folder.split(os.sep)[-2])
        timeout_flag = True
        while timeout_flag:
            try:
                driver.get(url)
                css_selector = "a.unnamed-link[href*='bioId']"
                time.sleep(3)
                barcode = driver.find_element_by_css_selector(css_selector).text
                timeout_flag = False
            except TimeoutException:
                driver.quit()
                driver = webdriver.Firefox(options=driver_options)
                driver.maximize_window()
        dataset["barcode"] = barcode
        new_dataset = new_dataset.append(dataset)
    driver.quit()
    new_dataset = new_dataset.set_index("barcode")
    return new_dataset
