from dataset.methylation450 import filter_cpg_islands
from dataset.methylation450 import create_methylation_dataset
import pickle

# First test to create a training dataset for the autoencoder

islands, num_files = filter_cpg_islands(["../data/lung_methylation_450"])
# islands, num_files = filter_cpg_islands(["../data/breast_methylation_450", "../data/lung_methylation_450"])
pickle.dump(islands, open("../data/lung_methylation_450.pkl", "wb"))
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
print(cpg_islands)
print(count)
print(len(cpg_islands))

dataset = create_methylation_dataset(["../data/lung_methylation_450"], cpg_islands)
pickle.dump(dataset, open("../data/lung_methylation_450_pd.pkl", "wb"))
dataset.to_csv("../data/lung_methylation_450_pd.csv", sep='\t')
