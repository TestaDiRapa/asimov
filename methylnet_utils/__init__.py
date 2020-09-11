from dataset import identify_breast_cancer_subtype
from dataset.methylation450 import folder_generator
import pandas as pd
import pickle


def generate_subtype_methylation_array(clinical_folder, methylation_dataset, output_filename):
    subtypes = dict()
    for file in folder_generator(clinical_folder, r'xml$'):
        subject_barcode = file.split('.')[-2]
        subtypes[subject_barcode] = identify_breast_cancer_subtype(file)

    beta = pickle.load(open(methylation_dataset, "rb"))
    barcodes = list(beta.index.values)
    pheno = pd.DataFrame(index=barcodes, columns=["subtype"])
    for barcode in barcodes:
        if barcode.split('-')[3][:2] == "11":
            pheno.loc[barcode] = "control"
        else:
            subject_barcode = '-'.join(barcode.split('-')[:3])
            pheno.loc[barcode] = subtypes[subject_barcode]
    pheno = pheno.dropna()
    pickle.dump({"beta": beta, "pheno": pheno}, open(output_filename, "wb"))
