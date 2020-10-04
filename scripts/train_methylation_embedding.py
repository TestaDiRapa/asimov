from dataset.mirna_exp import get_interactions_over_threshold
from methylnet_utils import split_methylation_array_by_pheno
from models import methylation_array_kcv
from models.autoencoders import Giskard, ShallowConvolutionalAE, DeepFullyConnectedAE, DeepConvolutionalAE
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import Jander
from models.generators import AutoencoderGenerator, MethylationArrayGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pickle
import numpy as np
import os


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


logfile_name = "../data/methylation_embedding.tsv"
if not os.path.exists(logfile_name):
    with open(logfile_name, "w") as logfile:
        fields = ["MODEL", "ACCURACY_EVALUATION_METHOD", "LATENT_DIMENSION", "VALIDATION_ACCURACY", "TEST_ACCURACY",
                  '\n']
        logfile.write('\t'.join(fields))

# Define the latent dimension
ld = 200

# Load the dataset and filter the CpG islands through a filtering process
dataset = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
"""
over_rate_genes = get_interactions_over_threshold(0.9, True)
genes_cpg = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
final_cpg = set()
for gene in over_rate_genes:
    if gene in genes_cpg:
        final_cpg = final_cpg.union(genes_cpg[gene])
index = set(dataset["beta"].columns.values)
final_cpg = final_cpg.intersection(index)
dataset["beta"] = dataset["beta"][final_cpg]
"""

pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')
islands = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
final_cpg = set()
for gene in pam50_genes:
    if gene in islands:
        final_cpg = final_cpg.union(islands[gene])
index = set(dataset["beta"].columns.values)
final_cpg = final_cpg.intersection(index)
dataset["beta"] = dataset["beta"][final_cpg]

# Generation of training and validation set
val_size = int(dataset["beta"].shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset["beta"].iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset["beta"].iloc[val_size:, :])

# Autoencoder training
methylation_encoder = DeepConvolutionalAE(dataset["beta"].shape[1], latent_dimension=ld,
                              model_serialization_path="../trained_models",
                              model_name="methylation_autoencoder")
methylation_encoder.fit(training_set, validation_set, 500,
                        callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

# Creating an embedded representation of the methylation data
methylation_dataset = methylation_encoder.encode_methylation_array(dataset)
# pickle.dump(methylation_dataset, open("../data/methylation_embedded_pam.pkl", "wb"))

# Just a check on ground truth
gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
gt_index = list(gt_check.index.values)
to_remove = list()
for index, row in methylation_dataset["pheno"].iterrows():
    if row["subtype"] != "control":
        barcode = "-".join(index.split("-")[:3])
        if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
            methylation_dataset["pheno"].at[index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
        else:
            to_remove.append(index)

methylation_dataset["beta"] = methylation_dataset["beta"].drop(to_remove)
methylation_dataset["pheno"] = methylation_dataset["pheno"].drop(to_remove)

"""
# Classification with ML and DL models
params = {"input_shape": methylation_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.2, "output_shape": len(methylation_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(methylation_dataset,
                                          Daneel,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

print("Validation accuracy: {} - Test accuracy: {}".format(val_res, test_res))
svm_val, svm_test = benchmark_svm(methylation_dataset, "subtype")
print("SVM validation accuracy: {} - SVM test accuracy: {}".format(svm_val, svm_test))
knn_val, knn_test = benchmark_knn(methylation_dataset, "subtype")
print("KNN validation accuracy: {} - KNN test accuracy: {}".format(knn_val, knn_test))
rf_val, rf_test = benchmark_rf(methylation_dataset, "subtype")
print("RF validation accuracy: {} - RF test accuracy: {}".format(rf_val, rf_test))


with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format(str(methylation_encoder), "MLP", ld, val_res, test_res))
    logfile.write(base.format(str(methylation_encoder), "SVM", ld, svm_val, svm_test))
    logfile.write(base.format(str(methylation_encoder), "KNN", ld, knn_val, knn_test))
    logfile.write(base.format(str(methylation_encoder), "RF", ld, rf_val, rf_test))
"""
stats = {
    "daneel": [],
    "svm": [],
    "knn": [],
    "rf": [],
}
for i in range(10):
    training_set, validation_set, test_set = \
        split_methylation_array_by_pheno(methylation_dataset, "subtype", val_rate=0.1, test_rate=0.1)

    # DNN - Daneel
    model, acc = train_dnn_classifier(Jander, training_set, validation_set, test_set)
    stats["daneel"].append(acc)

    # SVM
    model, acc = train_ml_classifier(training_set, test_set, SVC, {"C": 1, "kernel": "rbf"},
                                     validation_set)
    stats["svm"].append(acc)

    # KNN
    model, acc = train_ml_classifier(training_set, test_set, KNeighborsClassifier,
                                     {"n_neighbors": 75}, validation_set)
    stats["knn"].append(acc)

    # RF
    model, acc = train_ml_classifier(training_set, test_set, RandomForestClassifier,
                                     {"n_estimators": 2000, "max_features": "auto"}, validation_set)
    stats["rf"].append(acc)

for omic, accuracies in stats.items():
    print(omic, np.mean(accuracies))
