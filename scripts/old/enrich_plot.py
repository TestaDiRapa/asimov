from colour import Color
from enrichr import enrichr_query
import matplotlib.pyplot as plt
import numpy as np
import pickle

genes = set(pickle.load(open("../data/pam50_mrnas.pkl", "rb")))
pam50_genes = set(open("../data/PAM50_genes.txt").read().split('\n'))

library = "GO_Biological_Process_2018"
library = "KEGG_2019_Human"
data = enrichr_query(list(genes), library)
p_values = list()
enrichment = list()
cut = 10
start_colour = Color("turquoise")
colors = list(start_colour.range_to(Color("deepskyblue"), cut))
colors = [color.rgb for color in colors]

plt.figure(figsize=(10, 10), constrained_layout=True)
for result in data[library][::-1]:
    enrichment.append(result[1])
    p_values.append(-np.log10(result[2]))
enrichment = enrichment[-cut:]
p_values = p_values[-cut:]
y_pos = np.arange(len(enrichment))
print(y_pos)
print(p_values)
plt.barh(y_pos, p_values, color=colors)
plt.yticks(y_pos, enrichment)
plt.xlabel("-log10(p-value)")
plt.title("Top 10 KEGG pathways related to the selected genes")
plt.show()
