import pandas as pd
import pickle

pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')
pam50_ENSG = open("../data/PAM50_ENSG.txt").read().split('\n')
gene_to_cpgs = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
mirna_to_genes = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
ghr_data = pd.read_csv("../data/ghr_genes.txt", sep='\t')
# CpGs selection
pam50_cpgs = set()
for gene in pam50_genes:
    if gene in gene_to_cpgs:
        pam50_cpgs = pam50_cpgs.union(gene_to_cpgs[gene])
# miRNA selection
pam50_mirnas = list()
for mirna, genes in mirna_to_genes.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)
# mRNA selection
pam50_mrnas = list()
for m in pam50_mirnas:
    for gene in mirna_to_genes[m]:
        if gene[2] > 0.8:
            pam50_mrnas.append(gene[0])

methylation_encoder_dataset = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
index = set(methylation_encoder_dataset["beta"].columns.values)
pam50_cpgs = index.intersection(pam50_cpgs)
methylation_encoder_dataset = methylation_encoder_dataset["beta"][pam50_cpgs]
methylation_encoder_dataset = methylation_encoder_dataset.rename(index=lambda b: "-".join(b.split("-")[:4]))
index_1 = list(methylation_encoder_dataset.index.drop_duplicates(keep=False).values)
methylation_encoder_dataset = methylation_encoder_dataset.loc[index_1]

mrna_encoder_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))["beta"]
mrna_encoder_dataset = mrna_encoder_dataset.rename(columns=lambda g: g.split('.')[0])
mrna_encoder_dataset = mrna_encoder_dataset[pam50_mrnas]
mrna_encoder_dataset = mrna_encoder_dataset.rename(index=lambda b: "-".join(b.split("-")[:4]))
index_2 = list(mrna_encoder_dataset.index.drop_duplicates(keep=False).values)
mrna_encoder_dataset = mrna_encoder_dataset.loc[index_2]

mirna_encoder_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))["beta"]
mirna_encoder_dataset = mirna_encoder_dataset[pam50_mirnas]
mirna_encoder_dataset = mirna_encoder_dataset.rename(index=lambda b: "-".join(b.split("-")[:4]))
index_3 = list(mirna_encoder_dataset.index.drop_duplicates(keep=False).values)
mirna_encoder_dataset = mirna_encoder_dataset.loc[index_3]

result = pd.concat([methylation_encoder_dataset, mirna_encoder_dataset, mrna_encoder_dataset], axis=1, join='inner')
print(result.dropna())
