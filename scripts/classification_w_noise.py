from methylnet_utils import merge_methylation_arrays, split_methylation_array_by_pheno
from models.autoencoders import MiRNAEncoder
from models.classifiers import MOLIClassifier, PAMClassifier
from models.generators import AutoencoderGenerator, MethylationArrayGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import pickle


# SOME FUNCTIONS
def train_autoencoder(methylation_array, latent_dimension):
    val_size = int(methylation_array["beta"].shape[0] * 0.1)
    validation_set = AutoencoderGenerator(methylation_array["beta"].iloc[:val_size, :])
    training_set = AutoencoderGenerator(methylation_array["beta"].iloc[val_size:, :])

    # Autoencoder training
    encoder = MiRNAEncoder(methylation_array["beta"].shape[1], latent_dimension=latent_dimension,
                           model_serialization_path="../data/models/")
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


def train_dnn_classifier(classifier, training_set, validation_set, test_set):
    params = {"input_shape": training_set["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
              "dropout_rate": 0.3, "output_shape": len(training_set["pheno"]["subtype"].unique())}
    model = classifier(**params)
    model.fit(MethylationArrayGenerator(training_set, "subtype"),
              MethylationArrayGenerator(validation_set, "subtype"),
              500,
              verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])
    test_accuracy = model.evaluate(test_set["beta"].to_numpy(),
                                   pd.get_dummies(test_set["pheno"]["subtype"]).to_numpy())
    return model, test_accuracy


def train_ml_classifier(training_set, test_set, model, params, validation_set=None):
    if validation_set is not None:
        training_set["beta"] = training_set["beta"].append(validation_set["beta"])
        training_set["pheno"] = training_set["pheno"].append(validation_set["pheno"])
    classifier = model(**params)
    classifier.fit(training_set["beta"], training_set["pheno"].values.ravel())
    return classifier, classifier.score(test_set["beta"], test_set["pheno"].values.ravel())


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
methylation_embedded = methylation_encoder.encode_methylation_array(methylation_dataset)
# Training the mRNA autoencoder
mrna_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])
# mrna_dataset["beta"] = mrna_dataset["beta"][pam50_mrnas]
mrna_encoder = train_autoencoder(mrna_dataset, mrna_latent_dimension)
mrna_dataset = correct_labels(mrna_dataset)
mrna_embedded = mrna_encoder.encode_methylation_array(mrna_dataset)
# Training the miRNA autoencoder
mirna_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_dataset["beta"] = mirna_dataset["beta"][pam50_mirnas]
mirna_encoder = train_autoencoder(mirna_dataset, mirna_latent_dimension)
mirna_dataset = correct_labels(mirna_dataset)
mirna_embedded = mirna_encoder.encode_methylation_array(mirna_dataset)
# Combined dataset
combined_dataset = merge_methylation_arrays(methylation_dataset, mrna_dataset, mirna_dataset)
combined_embedded = merge_methylation_arrays(methylation_embedded, mrna_embedded, mirna_embedded)

# PART 3
# Training the different classifiers
stats = {
    "base": {
        "methylation": {
            "pam": [],
            "moli": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
        "mrna": {
            "pam": [],
            "moli": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
        "mirna": {
            "pam": [],
            "moli": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
        "combined": {
            "pam": [],
            "moli": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
    }
}

methylation_barcodes, mrna_barcodes, mirna_barcodes, combined_barcodes = [], [], [], []
for i in range(10):
    # Training models that take as input only methylation
    methylation_train, methylation_val, methylation_test = \
        split_methylation_array_by_pheno(methylation_embedded, "subtype", val_rate=0.1, test_rate=0.1)
    methylation_barcodes = list(methylation_test["pheno"].index.values)
    # DNN - PAM
    pam_methylation, dnn_acc = train_dnn_classifier(PAMClassifier,methylation_train, methylation_val, methylation_test)
    stats["base"]["methylation"]["dnn"].append(dnn_acc)
    # SVM
    svm_methylation, svm_acc = train_ml_classifier(methylation_train, methylation_test, SVC, {"C": 1, "kernel": "rbf"},
                                                   methylation_val)
    stats["base"]["methylation"]["svm"].append(svm_acc)
    # KNN
    knn_methylation, knn_acc = train_ml_classifier(methylation_train, methylation_test, KNeighborsClassifier,
                                                   {"n_neighbors": 75}, methylation_val)
    stats["base"]["methylation"]["knn"].append(knn_acc)
    # RF
    rf_methylation, rf_acc = train_ml_classifier(methylation_train, methylation_test, RandomForestClassifier,
                                                 {"n_estimators": 2000, "max_features": "auto"}, methylation_val)
    stats["base"]["methylation"]["rf"].append(rf_acc)

    # Training models that take as input only mRNA expression
    mrna_train, mrna_val, mrna_test = \
        split_methylation_array_by_pheno(mrna_embedded, "subtype", val_rate=0.1, test_rate=0.1)
    mrna_barcodes = list(mrna_test["pheno"].index.values)
    # DNN
    dnn_mrna, dnn_acc = train_dnn_classifier(mrna_train, mrna_val, mrna_test)
    stats["base"]["mrna"]["dnn"].append(dnn_acc)
    # SVM
    svm_mrna, svm_acc = train_ml_classifier(mrna_train, mrna_test, SVC, {"C": 1, "kernel": "rbf"}, mrna_val)
    stats["base"]["mrna"]["svm"].append(svm_acc)
    # KNN
    knn_mrna, knn_acc = train_ml_classifier(mrna_train, mrna_test, KNeighborsClassifier, {"n_neighbors": 75}, mrna_val)
    stats["base"]["mrna"]["knn"].append(knn_acc)
    # RF
    rf_mrna, rf_acc = train_ml_classifier(mrna_train, mrna_test, RandomForestClassifier,
                                          {"n_estimators": 2000, "max_features": "auto"}, mrna_val)
    stats["base"]["mrna"]["rf"].append(rf_acc)

    # Training models that take as input only miRNA expression
    mirna_train, mirna_val, mirna_test = \
        split_methylation_array_by_pheno(mirna_embedded, "subtype", val_rate=0.1, test_rate=0.1)
    mirna_barcodes = list(mirna_test["pheno"].index.values)
    # DNN
    dnn_mirna, dnn_acc = train_dnn_classifier(mirna_train, mirna_val, mirna_test)
    stats["base"]["mirna"]["dnn"].append(dnn_acc)
    # SVM
    svm_mirna, svm_acc = train_ml_classifier(mirna_train, mirna_test, SVC, {"C": 1, "kernel": "rbf"}, mirna_val)
    stats["base"]["mirna"]["svm"].append(svm_acc)
    # KNN
    knn_mirna, knn_acc = train_ml_classifier(mirna_train, mirna_test, KNeighborsClassifier, {"n_neighbors": 75},
                                             mirna_val)
    stats["base"]["mirna"]["knn"].append(knn_acc)
    # RF
    rf_mirna, rf_acc = train_ml_classifier(mirna_train, mirna_test, RandomForestClassifier,
                                           {"n_estimators": 2000, "max_features": "auto"}, mirna_val)
    stats["base"]["mirna"]["rf"].append(rf_acc)

    # Training the combined model
    combined_train, combined_val, combined_test = \
        split_methylation_array_by_pheno(combined_embedded, "subtype", val_rate=0.1, test_rate=0.1)
    combined_barcodes = list(combined_test["pheno"].index.values)
    # DNN
    dnn_combined, dnn_acc = train_dnn_classifier(combined_train, combined_val, combined_test)
    stats["base"]["combined"]["dnn"].append(dnn_acc)
    # SVM
    svm_combined, svm_acc = train_ml_classifier(combined_train, combined_test, SVC, {"C": 1, "kernel": "rbf"},
                                                combined_val)
    stats["base"]["combined"]["svm"].append(svm_acc)
    # KNN
    knn_combined, knn_acc = train_ml_classifier(combined_train, combined_test, KNeighborsClassifier,
                                                {"n_neighbors": 75}, combined_val)
    stats["base"]["combined"]["knn"].append(knn_acc)
    # RF
    rf_combined, rf_acc = train_ml_classifier(combined_train, combined_test, RandomForestClassifier,
                                              {"n_estimators": 2000, "max_features": "auto"}, combined_val)
    stats["base"]["combined"]["rf"].append(rf_acc)

for omic, omic_stats in stats["base"].items():
    for c, scores in omic_stats.items():
        print("{} {} mean acc: {} - std {}".format(omic, c, np.mean(scores), np.std(scores)))
