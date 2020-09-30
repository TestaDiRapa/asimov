from models.autoencoders import MiRNAEncoder
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pickle


# SOME FUNCTIONS
def train_autoencoder(methylation_array, latent_dimension):
    val_size = int(methylation_array["beta"].shape[0] * 0.1)
    validation_set = AutoencoderGenerator(methylation_array["beta"].iloc[:val_size, :])
    training_set = AutoencoderGenerator(methylation_array["beta"].iloc[val_size:, :])

    # Autoencoder training
    encoder = MiRNAEncoder(methylation_array["beta"].shape[1], latent_dimension=latent_dimension,
                           model_serialization_path="../trained_models",
                           model_name="methylation_autoencoder")
    encoder.fit(training_set, validation_set, 500,
                callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
    return encoder


def correct_labels(methylation_array):
    gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
    gt_index = list(gt_check.index.values)
    to_remove = list()
    for pheno_index, row in methylation_array["pheno"].iterrows():
        if row["subtype"] != "control":
            barcode = "-".join(pheno_index.split("-")[:3])
            if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
                methylation_array["pheno"].at[pheno_index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
            else:
                to_remove.append(pheno_index)

    methylation_array["beta"] = methylation_array["beta"].drop(to_remove)
    methylation_array["pheno"] = methylation_array["pheno"].drop(to_remove)
    return methylation_array


# Defining parameters for training
methylation = True
mirna = True
mrna = True
methylation_latent_dimension = 200
mirna_latent_dimension = 200
mrna_latent_dimension = 200
dataset_list = []

# PART 1
# CpG, mRNA and miRNA selection using PAM50 genes and first interactors
pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')
pam50_ENSG = open("../data/PAM50_ENSG.txt").read().split('\n')
gene_to_cpgs = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
mirna_to_genes = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
# CpGs selection
pam50_cpgs = set()
for gene in pam50_genes:
    if gene in gene_to_cpgs:
        pam50_cpgs = pam50_cpgs.union(gene_to_cpgs[gene])
# mRNA selection
pam50_mirnas = list()
for mirna, genes in mirna_to_genes.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)
pam50_mrnas = list()
for m in pam50_mirnas:
    for gene in mirna_to_genes[m]:
        if gene[2] > 0.8:
            pam50_mrnas.append(gene[0])
# miRNA selection
pam50_mirnas = list()
for mirna, genes in mirna_to_genes.items():
    for gene in genes:
        if gene[1] in pam50_genes:
            pam50_mirnas.append(mirna)

# PART 2
# Training the different autoencoders
# Training the methylation autoencoder
methylation_dataset = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
index = set(methylation_dataset["beta"].columns.values)
pam50_cpgs = index.intersection(pam50_cpgs)
methylation_dataset["beta"] = methylation_dataset["beta"][pam50_cpgs]
methylation_encoder = train_autoencoder(methylation_dataset, methylation_latent_dimension)
methylation_dataset = correct_labels(methylation_dataset)
# Training the mRNA autoencoder
mrna_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])
mrna_dataset["beta"] = mrna_dataset["beta"][pam50_mrnas]
mrna_encoder = train_autoencoder(mrna_dataset, mrna_latent_dimension)
mrna_dataset = correct_labels(mrna_dataset)
# Training the miRNA autoencoder
mirna_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_dataset["beta"] = mirna_dataset["beta"][pam50_mirnas]
mirna_encoder = train_autoencoder(mirna_dataset, mirna_latent_dimension)
mirna_dataset = correct_labels(mirna_dataset)

# PART 3
# Training the different classifiers
methylation_embedded = methylation_encoder.encode_methylation_array(methylation_dataset)
mrna_embedded = mrna_encoder.encode_methylation_array(mrna_dataset)
mirna_embedded = mirna_encoder.encode_methylation_array(mirna_dataset)

