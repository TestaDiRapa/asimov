from dataset.methylation450 import create_methylation_dataset
import pickle

islands = pickle.load(open("../data/pam50_cpg.pkl", "rb"))
dataset = create_methylation_dataset(["../data/other_methylation_exp"], islands)
print(dataset)
print(islands)
all_data = pickle.load(open("../data/methylation_exp_all.pkl", "rb"))
all_data = all_data.append(dataset)
print(all_data)
pickle.dump(all_data, open("../data/methylation_exp_all.pkl", "wb"))
