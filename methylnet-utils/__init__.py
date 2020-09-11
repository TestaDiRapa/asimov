from dataset import identify_breast_cancer_subtype
from dataset.methylation450 import folder_generator
import pandas as pd
import pickle


def generate_subtype_methylation_array(clinical_folder, methylation_dataset):
    subtypes = dict()
    for file in folder_generator(clinical_folder, r'^xml$'):
        subject_barcode = file.split('.')[-2]
        subtypes[subject_barcode] = identify_breast_cancer_subtype(file)

    beta = pd.read_csv(methylation_dataset, sep='\t', index_col="barcode")
    barcodes = list(beta.index.values)
    print(len(barcodes))
    pheno = pd.DataFrame(index=beta, columns=["subtype"])
    for barcode in barcodes:
        if barcode.split('-')[3][:2] == "11":
            pheno.loc[barcode] = "control"
        else:
            subject_barcode = '-'.join(barcode.split('-')[:3])
            pheno.loc[barcode] = subtypes[subject_barcode]
    print(pheno)
    pickle.dump({"beta": beta, "pheno":pheno}, open("breast_methylation_450_ma.pkl", "wb"))
