from dataset import feature_counter
from dataset.methylation450 import create_methylation_dataset
from methylnet_utils import generate_subtype_methylation_array
import pandas as pd
import pickle
"""
pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')
pam50_ENSG = open("../data/PAM50_ENSG.txt").read().split('\n')
gene_to_cpgs = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
mirna_to_genes = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
# CpGs selection
pam50_cpgs = set()
for gene in pam50_genes:
    if gene in gene_to_cpgs:
        pam50_cpgs = pam50_cpgs.union(gene_to_cpgs[gene])

path = "../data/breast_methylation_27K/0a7e4016-a7e0-4e8f-a27b-735c8c8671d9/jhu-usc.edu_BRCA.HumanMethylation27.3.lvl-3.TCGA-BH-A0C7-01B-11D-A112-05.gdc_hg38.txt"
dataset = pd.read_csv(path, sep='\t', na_values="NA", index_col=0)

cpg_27k = set()
for index, row in dataset.iterrows():
    genes = row["Gene_Symbol"].split(";")
    for g in pam50_genes:
        if g in genes:
            cpg_27k.add(index)

methylation_dataset = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
index_ = set(methylation_dataset["beta"].columns.values)

print(len(pam50_cpgs))
print(len(cpg_27k.intersection(index_)))
pickle.dump(cpg_27k, open("../data/cpg_list_27k.pkl", "wb"))
cpg_27k = cpg_27k.intersection(index_)
"""
islands, num_files = feature_counter(["../../data/raw/breast_methylation_27K"], r'^jhu-usc\..+txt$',
                                     "Composite Element REF", ["Composite Element REF", "Beta_value"])
# islands, num_files = filter_cpg_islands(["../data/breast_methylation_450", "../data/lung_methylation_450"])
# pickle.dump(islands, open("../data/lung_methylation_450.pkl", "wb"))
# islands, num_files = pickle.load(open("../data/breast_methylation_450.pkl", "rb")), 893  # +920
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

cpg_islands = []

for cpg, num_instances in islands.items():
    percentage = num_instances*100/num_files
    if percentage == 100:
        count["100"] += 1
        if cpg[:2] == "cg":
            cpg_islands.append(cpg)
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

dataset = create_methylation_dataset("../../data/raw/breast_methylation_27K", cpg_islands)
ma = generate_subtype_methylation_array("../../data/raw/breast_clinical", dataset)
pickle.dump(ma, open("../../data/breast/methylation/breast_methylation_27k.pkl", "wb"))

deep_cc_brca = pd.read_csv("../../data/features/gt/deep_cc_brca.csv", index_col=1)
deep_cc_brca = deep_cc_brca.rename(index=lambda b: "-".join(b.split("-")[:4])[:-1])
deep_cc_brca = deep_cc_brca.dropna()
deep_cc_index = set(deep_cc_brca.index.values)


gt_check = pd.read_csv("../../data/features/gt/brca_tcga_pub_clinical_data.tsv",
                       sep="\t", na_filter=False, index_col="Sample ID")
gt_index = set(gt_check.index.values)

dataset = dataset.rename(index=lambda b: "-".join(b.split("-")[:4])[:-1])
index_ = set(dataset.index.values)

print("Deep CC overlap: {}".format(len(deep_cc_index.intersection(index_))))
print("GT overlap: {}".format(len(gt_index.intersection(index_))))

to_remove = list()
for pheno_index, row in ma["pheno"].iterrows():
    if row["subtype"] != "control":
        barcode = "-".join(pheno_index.split("-")[:4])[:-1]
        if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
            ma["pheno"].at[pheno_index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
        else:
            to_remove.append(pheno_index)

ma["beta"] = ma["beta"].drop(to_remove)
ma["pheno"] = ma["pheno"].drop(to_remove)

pickle.dump(ma, open("../../data/breast/methylation/breast_methylation_27k.pkl", "wb"))
print(ma)
