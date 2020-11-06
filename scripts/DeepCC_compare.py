import pandas as pd
import pickle


def correct_labels(methylation_array):
    gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
    gt_index = list(gt_check.index.values)
    to_remove = list()
    for pheno_index, row in methylation_array["pheno"].iterrows():
        if row["subtype"] != "control":
            barcode = "-".join(pheno_index.split("-")[:3])
            if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
                methylation_array["pheno"].at[pheno_index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
            else:
                to_remove.append(pheno_index)

    methylation_array["beta"] = methylation_array["beta"].drop(to_remove)
    methylation_array["pheno"] = methylation_array["pheno"].drop(to_remove)
    return methylation_array


met = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
met["pheno"] = met["pheno"].rename(index=lambda b: "-".join(b.split("-")[:4]))
met_corr = correct_labels(pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb")))
met_corr["pheno"] = met_corr["pheno"].rename(index=lambda b: "-".join(b.split("-")[:4]))
deep_cc_brca = pd.read_csv("../data/deep_cc_brca.csv", index_col=1).rename(index=lambda b: "-".join(b.split("-")[:4]))
deep_cc_brca = deep_cc_brca.dropna()
deep_cc_brca = deep_cc_brca.replace(["LumB", "LumA", "control", "Basal", "NA", "Her2"],
                                    ["Luminal B", "Luminal A", "control", "Basal-like", "NA", "HER2-enriched"])
met_index = set(met["pheno"].index.values)
corr_index = set(met_corr["pheno"].index.values)
cc_index = set(deep_cc_brca.index.values)
i = cc_index.intersection(corr_index)
df1 = met_corr["pheno"].loc[i]
df2 = deep_cc_brca.loc[i]
df2 = df2.loc[:, ["DeepCC"]]
df2.rename(columns={"DeepCC": "subtype"})

count_tmp = dict()
for el in df2["DeepCC"].unique():
    count_tmp[el] = 0

for index, row in df2.iterrows():
    count_tmp[row["DeepCC"]] += 1

print(df2)
print(df2.values.ravel())