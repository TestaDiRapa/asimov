from dataset import folder_generator, identify_breast_cancer_subtype
import pandas as pd

translation = {
    "Basal-like": "TNBS",
    "HER2-enriched": "HER2+",
    "Luminal A": "LuminalA",
    "Luminal B": "LuminalB",
    "NA": "Unclear",
    "Normal-like": "Normal-like"
}

count_types = {
    "Basal-like": 0,
    "HER2-enriched": 0,
    "Luminal A": 0,
    "Luminal B": 0,
    "NA": 0,
    "Normal-like": 0
}

subtypes = dict()
for file_ in folder_generator("../data/breast_clinical", r"xml$"):
    barcode = file_.split('.')[-2]
    subtypes[barcode] = identify_breast_cancer_subtype(file_)
    
check_data = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False)

count = 0
for index, row in check_data.iterrows():
    barcode = row["Patient ID"]
    subtype = row["PAM50 subtype"]
    if barcode in subtypes:
        if subtypes[barcode] == translation[subtype]:
            count_types[subtype] += 1
        else:
            pass
            # print("CHECK: {}, INFERRED: {}".format(subtype, subtypes[barcode]))
    else:
        print("No match")
print(count)
print(count_types)
print(check_data)

