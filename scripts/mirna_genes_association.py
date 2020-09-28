from dataset.mirna_exp import mirna_genes_interaction
import pandas as pd
import pickle

dataset = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
"""
interactions = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
for mirna in dataset.columns:
    if mirna not in interactions:
        print(mirna)
        interactions[mirna] = mirna_genes_interaction(mirna)
        print(interactions[mirna])
        pickle.dump(interactions, open("../data/mirna_genes_interaction.pkl", "wb"))
"""
stats = {
    "total": set(),
    ">90": set(),
    ">80": set(),
    ">70": set(),
    ">60": set(),
    ">50": set(),
    ">40": set(),
    ">30": set(),
    ">20": set(),
    ">10": set(),
}
for mirna in dataset:
    for gene_stat in dataset[mirna]:
        stats["total"].add(gene_stat[1])
        if gene_stat[2] >= 0.9:
            stats[">90"].add(gene_stat[1])
        if gene_stat[2] >= 0.8:
            stats[">80"].add(gene_stat[1])
        if gene_stat[2] >= 0.7:
            stats[">70"].add(gene_stat[1])
        if gene_stat[2] >= 0.6:
            stats[">60"].add(gene_stat[1])
        if gene_stat[2] >= 0.5:
            stats[">50"].add(gene_stat[1])
        if gene_stat[2] >= 0.4:
            stats[">40"].add(gene_stat[1])
        if gene_stat[2] >= 0.3:
            stats[">30"].add(gene_stat[1])
        if gene_stat[2] >= 0.2:
            stats[">20"].add(gene_stat[1])
        if gene_stat[2] >= 0.1:
            stats[">10"].add(gene_stat[1])

for key, value in stats.items():
    print(key, len(value))

gene_cpg = dict()
pickle.dump(gene_cpg, open("../data/genes_cpg_interaction.pkl", "wb"))

met_df = pd.read_csv("../data/methylation_example.txt", sep="\t", na_values="NA").dropna()
for index, row in met_df.iterrows():
    for gene in row["Gene_Symbol"].split(";"):
        if gene != ".":
            if gene not in gene_cpg:
                gene_cpg[gene] = set()
            gene_cpg[gene].add(row["Composite Element REF"])

final_cpg = set()
count = 0
for gene in stats[">90"]:
    if gene not in gene_cpg:
        count += 1
    else:
        final_cpg = final_cpg.union(gene_cpg[gene])

print(final_cpg)
print("Rejected:", count)
print(len(final_cpg))
