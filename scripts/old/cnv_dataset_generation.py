from dataset import folder_generator
from dataset import merge_tsv_to_pandas
from methylnet_utils import generate_subtype_methylation_array
import os
import pandas as pd
import pickle


genes = dict()
num_files = 0
for path in folder_generator("../data/breast_cnv", r'tsv$'):
    num_files += 1
    dataset = pd.read_csv(path, sep='\t', index_col=0, na_values="NA")
    dataset = dataset.dropna()
    for gene in dataset.index.values:
        if gene in genes:
            genes[gene] += 1
        else:
            genes[gene] = 1

count = {
    "100": 0,
    "99-90": 0,
    "89-80": 0,
    "79-70": 0,
    "69-60": 0,
    "59-50": 0,
    "49-40": 0,
    "39-30": 0,
    "29-20": 0,
    "19-10": 0,
    "9-0": 0
}

final_genes = []

for gene, num_instances in genes.items():
    percentage = num_instances*100/num_files
    if percentage == 100:
        count["100"] += 1
        final_genes.append(gene)
    elif percentage >= 90:
        count["99-90"] += 1
    elif percentage >= 80:
        count["89-80"] += 1
    elif percentage >= 70:
        count["79-70"] += 1
    elif percentage >= 60:
        count["69-60"] += 1
    elif percentage >= 50:
        count["59-50"] += 1
    elif percentage >= 40:
        count["49-40"] += 1
    elif percentage >= 30:
        count["39-30"] += 1
    elif percentage >= 20:
        count["29-20"] += 1
    elif percentage >= 10:
        count["19-10"] += 1
    else:
        count["9-0"] += 1

print(num_files, len(genes.keys()))
cnv_folders = [os.path.join("..", "data", "breast_cnv")]
dataset = merge_tsv_to_pandas(cnv_folders, final_genes, "copy_number", r"tsv$")
print(dataset)
pickle.dump(dataset, open("../data/cnv_exp.pkl", "wb"))

# dataset = pickle.load(open("../data/mrna_exp.pkl", "rb"))

met_dataset = generate_subtype_methylation_array("../data/breast_clinical", dataset)
pickle.dump(met_dataset, open("../data/cnv_exp_ma.pkl", "wb"))
print("Finished")
