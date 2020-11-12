from colour import Color
from enrichr import enrichr_query
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("../data/mirna_pathways.csv")
data = data.iloc[:10,]
data["p-value"] = -np.log10(data["p-value"])
enrichment, p_values = [], []
for index, row in data.iterrows():
    enrichment.append(row["KEGG pathway"])
    p_values.append(row["p-value"])
enrichment = enrichment[::-1]
p_values = p_values[::-1]
cut = 10
start_colour = Color("turquoise")
colors = list(start_colour.range_to(Color("deepskyblue"), cut))
colors = [color.rgb for color in colors]
plt.figure(figsize=(10, 10), constrained_layout=True)
y_pos = np.arange(cut)
print(y_pos)
print(len(p_values))
plt.barh(y_pos, p_values, color=colors)
plt.yticks(y_pos, enrichment)
plt.xlabel("-log10(p-value)")
plt.title("Top 10 KEGG pathways related to the selected genes")
plt.show()