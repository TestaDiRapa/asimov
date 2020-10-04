from methylnet_utils import merge_methylation_arrays, split_methylation_array_by_pheno
from models.autoencoders import Giskard
from models.classifiers import NeuralClassifier, ConvolutionalClassifier, Daneel, Jander, SmallClassifier, \
    SmallConvolutionalClassifier, MediumClassifier
from models.generators import AutoencoderGenerator, MethylationArrayGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


# SOME FUNCTIONS
def train_autoencoder(methylation_array, latent_dimension):
    val_size = int(methylation_array["beta"].shape[0] * 0.1)
    val_set = AutoencoderGenerator(methylation_array["beta"].iloc[:val_size, :])
    train_set = AutoencoderGenerator(methylation_array["beta"].iloc[val_size:, :])

    # Autoencoder training
    encoder = Giskard(methylation_array["beta"].shape[1], latent_dimension=latent_dimension,
                      model_serialization_path="../data/models/")
    encoder.fit(train_set, val_set, 500,
                callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
    return encoder


def correct_labels(methylation_array):
    gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
    gt_index = list(gt_check.index.values)
    to_remove = list()
    count_na = 0
    for pheno_index, row in methylation_array["pheno"].iterrows():
        if row["subtype"] != "control":
            barcode = "-".join(pheno_index.split("-")[:3])
            if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
                if gt_check.loc[barcode]["PAM50 subtype"] != "NA":
                    methylation_array["pheno"].at[pheno_index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
                elif gt_check.loc[barcode]["PAM50 subtype"] == "NA" and count_na < 150:
                    count_na += 1
                    methylation_array["pheno"].at[pheno_index, "subtype"] = "NA"
                else:
                    to_remove.append(pheno_index)
            else:
                to_remove.append(pheno_index)

    methylation_array["beta"] = methylation_array["beta"].drop(to_remove)
    methylation_array["pheno"] = methylation_array["pheno"].drop(to_remove)
    return methylation_array


def train_dnn_classifier(classifier, train_set, val_set, test_data):
    params = {"input_shape": train_set["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
              "dropout_rate": 0.3, "output_shape": len(train_set["pheno"]["subtype"].unique())}
    model = classifier(**params)
    model.fit(MethylationArrayGenerator(train_set, "subtype"),
              MethylationArrayGenerator(val_set, "subtype"),
              500,
              verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])
    test_accuracy = model.evaluate(test_data["beta"].to_numpy(),
                                   pd.get_dummies(test_data["pheno"]["subtype"]).to_numpy())
    return model, test_accuracy


def train_ml_classifier(train_set, test_data, model, params, val_set=None):
    if val_set is not None:
        train_set["beta"] = train_set["beta"].append(val_set["beta"])
        train_set["pheno"] = train_set["pheno"].append(val_set["pheno"])
    classifier = model(**params)
    classifier.fit(train_set["beta"], train_set["pheno"].values.ravel())
    return classifier, classifier.score(test_data["beta"], test_data["pheno"].values.ravel())


def slice_methylation_array_to_merge(methylation_array, b):
    return {
        "pheno": methylation_array["pheno"].rename(index=lambda barcode: "-".join(barcode.split("-")[:4])).loc[b],
        "beta": methylation_array["beta"].rename(index=lambda barcode: "-".join(barcode.split("-")[:4])).loc[b]
    }


def plot_single_omic_change(acc_scores, ml_method, omic_type, single=True):
    x, y = [], []
    x.append(0)
    y.append(np.mean(acc_scores["base"][omic_type][ml_method]))
    if single:
        scope = "single"
    else:
        scope = "combined"
    for std, a in acc_scores[scope][omic_type][ml_method].items():
        x.append(std)
        y.append(a)

    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    plt.title("Accuracy of {} adding noise to {}".format(ml_method, omic_type))
    plt.xlabel("Variance of the noise")
    plt.ylabel("Accuracy")
    plt.savefig("../data/results/{}_{}_{}.png".format(ml_method, omic_type, scope))


def plot_against_combined(acc_scores, ml_method, omic_type):
    x_single, y_single = [], []
    x_combined, y_combined = [], []
    x_single.append(0)
    y_single.append(np.mean(acc_scores["base"][omic_type][ml_method]))
    for std, a in acc_scores["single"][omic_type][ml_method].items():
        x_single.append(std)
        y_single.append(a)

    x_combined.append(0)
    y_combined.append(np.mean(acc_scores["base"]["combined"][ml_method]))
    for std, a in acc_scores["combined"][omic_type][ml_method].items():
        x_combined.append(std)
        y_combined.append(a)

    plt.figure(figsize=(10, 10))
    plt.plot(x_single, y_single, label="single")
    plt.plot(x_combined, y_combined, label="combined")
    plt.title("Accuracy of {} adding noise to {}".format(ml_method, omic_type))
    plt.xlabel("Variance of the noise")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("../data/results/{}_{}_single_vs_combined.png".format(ml_method, omic_type))


# Defining parameters for training
omics = ["methylation", "mrna", "mirna", "combined"]
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
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)

# PART 2
# Training the different autoencoders
# Training the methylation autoencoder
datasets = dict()
methylation_dataset = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
index = set(methylation_dataset["beta"].columns.values)
pam50_cpgs = index.intersection(pam50_cpgs)
methylation_dataset["beta"] = methylation_dataset["beta"][pam50_cpgs]
methylation_encoder = train_autoencoder(methylation_dataset, methylation_latent_dimension)
datasets["methylation"] = {
    "original": correct_labels(methylation_dataset),
    "embedded": methylation_encoder.encode_methylation_array(methylation_dataset)
}
# Training the mRNA autoencoder
mrna_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])
mrna_dataset["beta"] = mrna_dataset["beta"][pam50_mrnas]
mrna_encoder = train_autoencoder(mrna_dataset, mrna_latent_dimension)
datasets["mrna"] = {
    "original": correct_labels(mrna_dataset),
    "embedded": mrna_encoder.encode_methylation_array(mrna_dataset)
}
# Training the miRNA autoencoder
mirna_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_dataset["beta"] = mirna_dataset["beta"][pam50_mirnas]
mirna_encoder = train_autoencoder(mirna_dataset, mirna_latent_dimension)
datasets["mirna"] = {
    "original": correct_labels(mirna_dataset),
    "embedded": mirna_encoder.encode_methylation_array(mirna_dataset)
}
# Combined dataset
datasets["combined"] = {
    "embedded": merge_methylation_arrays(
        datasets["methylation"]["embedded"],
        datasets["mrna"]["embedded"],
        datasets["mirna"]["embedded"]
    )
}

encoders = {
    "methylation": methylation_encoder,
    "mrna": mrna_encoder,
    "mirna": mirna_encoder
}
print(datasets["combined"]["embedded"])
raise(Exception())
# PART 3
# Training the different classifiers
stats = {
    NeuralClassifier: [],
    ConvolutionalClassifier: [],
    Daneel: [],
    Jander: [],
    SmallClassifier: [],
    SmallConvolutionalClassifier: [],
    MediumClassifier: []
}

barcodes, models = dict(), dict(),
for i in range(10):
    training_set, validation_set, test_set = \
        split_methylation_array_by_pheno(datasets["combined"]["embedded"], "subtype", val_rate=0.1, test_rate=0.1)

    for model in stats.keys():
        _, acc = train_dnn_classifier(model, training_set, validation_set, test_set)
        stats[model].append(acc)


for omic, accuracies in stats.items():
    print(omic, np.mean(accuracies))

