import pandas as pd
import pickle

gt = pd.read_csv("../data/coadread_tcga_pub_clinical_data.tsv", sep='\t')
gt = gt.fillna("NA")
gt = gt[["Sample ID", "Gene Expression Subtype"]].set_index("Sample ID").rename(
    columns={"Gene Expression Subtype": "subtype"})
gt_index = set(gt.index.values)
met = pickle.load(open("../data/methylation_exp_crc.pkl", "rb"))
met = met.rename(index=lambda b: "-".join(b.split("-")[:4])[:-1])
met_index = set(met.index.values)
new_index = met_index.intersection(gt_index)
for m in met_index:
    if m.split("-")[3] == "11":
        new_index.add(m)
beta = met.loc[new_index]
new_index = list(beta.index.values)
pheno = pd.DataFrame(index=new_index, columns=["subtype"])
for barcode in new_index:
    if barcode.split("-")[3] == "11":
        pheno.at[barcode, "subtype"] = "control"
    else:
        pheno.at[barcode, "subtype"] = gt.loc[barcode, "subtype"]

pickle.dump({"beta": beta, "pheno": pheno}, open("../data/crc_methylation_ma.pkl", "wb"))
