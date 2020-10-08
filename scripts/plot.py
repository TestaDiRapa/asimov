from enrichr import enrichr_query
import matplotlib.pyplot as plt
import numpy as np
import pickle

noise_data = pickle.load(open("../data/noise_effect_on_genes.pkl", "rb"))
noise_avg = dict()
for gene, data in noise_data.items():
    noise_avg[gene] = np.mean(data, axis=0)
"""
plt.figure()
x = np.linspace(0.001, 100, num=50)
for gene, data in noise_avg.items():
    plt.plot(x, data, label=gene)
plt.legend(loc="best")
plt.show()
"""
worst_5 = [k for k, v in sorted(noise_avg.items(), key=lambda item: np.mean(item[1]))][:5]
plt.figure(figsize=(10, 10))
x = np.linspace(0.001, 100, num=50)
for gene in worst_5:
    plt.plot(x, noise_avg[gene], label=gene)
plt.legend(loc="best")
plt.show()

# data = enrichr_query(worst_5, "GO_Biological_Process_2018")
# print(data)
