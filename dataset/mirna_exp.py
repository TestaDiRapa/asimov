from dataset import folder_generator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
    count = 1
    for folder in folder_generator(mirna_folder, r'mirnas\.quantification\.txt$'):
        print(count)
        count += 1
        dataset = pd.read_csv(folder, sep='\t', na_values="NA", index_col=0)
        dataset = dataset[["reads_per_million_miRNA_mapped"]].dropna().loc[mirnas].T.reset_index().drop("index", axis=1)
        dataset.columns.name = None
        url = "https://portal.gdc.cancer.gov/files/{}".format(folder.split(os.sep)[-2])
        print(url)
        timeout_flag = True
        while timeout_flag:
            try:
                driver.get(url)
                print("GET OK")
                css_selector = "a.unnamed-link[href*='bioId']"
                # WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))
                time.sleep(3)
                print("WAIT OK")
                barcode = driver.find_element_by_css_selector(css_selector).text
                timeout_flag = False
                print("TRY OUT")
            except TimeoutException:
                print("TIMEOUT")
                driver.quit()
                driver = webdriver.Firefox(options=driver_options)
                driver.maximize_window()
            except Exception as e:
                print("Another exception occurred:", str(e))
                driver.quit()
                driver = webdriver.Firefox(options=driver_options)
                driver.maximize_window()
        print(barcode)
        dataset["barcode"] = barcode
        new_dataset = new_dataset.append(dataset)
    driver.quit()
    new_dataset = new_dataset.set_index("barcode")
    return new_dataset
