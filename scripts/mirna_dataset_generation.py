from dataset import feature_counter
from dataset import merge_tsv_to_pandas
# from methylnet_utils import generate_subtype_methylation_array
import os
import pickle

mirnas, num_files = feature_counter("../data/other_mirna_exp", r'mirnas\.quantification\.txt$',
                                    "miRNA_ID", ["reads_per_million_miRNA_mapped"])
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

final_mirnas = []

for cpg, num_instances in mirnas.items():
    percentage = num_instances*100/num_files
    if percentage == 100:
        count["100"] += 1
        final_mirnas.append(cpg)
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

print(num_files, len(mirnas.keys()))
dataset = merge_tsv_to_pandas(os.path.join("..", "data", "other_mirna_exp"), final_mirnas,
                              "reads_per_million_miRNA_mapped", r"mirnas\.quantification\.txt$", barcode=False)
print(dataset)
pickle.dump(dataset, open("../data/mirna_exp_all.pkl", "wb"))
# dataset = pickle.load(open("../data/mirna_exp.pkl", "rb"))
# met_dataset = generate_subtype_methylation_array("../data/breast_clinical", dataset)
# pickle.dump(met_dataset, open("../data/mirna_exp_ma.pkl", "wb"))
