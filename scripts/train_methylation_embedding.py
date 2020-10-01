from dataset.mirna_exp import get_interactions_over_threshold
from models import methylation_array_kcv
from models.autoencoders import MiRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier, MOLIClassifier
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pickle
import os

os.environ['PYTHONHASHSEED'] = '42'

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
methylation_encoder = MiRNAEncoder(dataset["beta"].shape[1], latent_dimension=ld,
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

# Classification with ML and DL models
params = {"input_shape": methylation_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.2, "output_shape": len(methylation_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(methylation_dataset,
                                          MOLIClassifier,
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
