import pandas as pd
import pickle

path = "../../data/raw/breast_methylation_450/0a04b3e2-cf44-41b0-bde4-d3cd17adf3ae/" \
       "jhu-usc.edu_BRCA.HumanMethylation450.11.lvl-3.TCGA-EW-A1P6-01A-11D-A145-05.gdc_hg38.txt"
reference = pd.read_csv(path, sep='\t', na_values="NA", index_col=0)

cpg_info = dict()
for index, row in reference.iterrows():
    types = set(row["Gene_Type"].split(';'))
    if '.' in types:
        types = set()
    genes = set(row["Gene_Symbol"].split(';'))
    if '.' in genes:
        genes = set()
    ft = row["Feature_Type"]
    if ft == '.':
        ft = None
    cpg_info[index] = {
        "types": types,
        "genes": genes,
        "feature_type": ft
        }

pickle.dump(cpg_info, open("../../data/features/cpg_info.pkl", "wb"))
