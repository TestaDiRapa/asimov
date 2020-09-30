from dataset import filter_expression_by_rate
from dataset.mirna_exp import get_interactions_over_threshold
from models import methylation_array_kcv
from models.autoencoders import MiRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
import pickle

logfile_name = "../data/mrna_embedding.tsv"
if not os.path.exists(logfile_name):
    with open(logfile_name, "w") as logfile:
        fields = ["MODEL", "ACCURACY_EVALUATION_METHOD", "LATENT_DIMENSION", "VALIDATION_ACCURACY", "TEST_ACCURACY",
                  '\n']
        logfile.write('\t'.join(fields))

# Define the latent dimension
ld = 200

# Load the dataset and filter the mRNA keeping only the ones that are non 0 for a certain rate of samples
dataset = pickle.load(open("../data/mrna_exp.pkl", "rb")).rename(columns=lambda g: g.split('.')[0])
# over_rate_mrna = filter_expression_by_rate(dataset, 0.9)
# over_rate_mrna = get_interactions_over_threshold(0.9, False)
# miRNA related to PAM50 genes
pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')
mirnas = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
pam50_mirnas = list()
for mirna, genes in mirnas.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)

# Genes related to miRNAs
over_rate_genes = list()
for m in pam50_mirnas:
    for gene in mirnas[m]:
        if gene[2] > 0.8:
            over_rate_genes.append(gene[0])

# dataset = pickle.load(open("../data/mrna_exp_all.pkl", "rb"))[over_rate_mrna]
dataset = dataset[over_rate_genes]

# Generation of training and validation set
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])

# Autoencoder training
mrna_encoder = MiRNAEncoder(dataset.shape[1], latent_dimension=ld, model_serialization_path="../data/models/")
mrna_encoder.fit(training_set, validation_set, 2000,
                 callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

# Creating an embedded representation of the mRNA methylation array
mrna_to_encode = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_to_encode["beta"] = mrna_to_encode["beta"].rename(columns=lambda g: g.split('.')[0])[over_rate_genes]
mrna_dataset = mrna_encoder.encode_methylation_array(mrna_to_encode)
pickle.dump(mrna_dataset, open("../data/mrna_embedded_pam.pkl", "wb"))

# Just a check on ground truth
gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
gt_index = list(gt_check.index.values)
to_remove = list()
for index, row in mrna_dataset["pheno"].iterrows():
    if row["subtype"] != "control":
        barcode = "-".join(index.split("-")[:3])
        if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
            mrna_dataset["pheno"].at[index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
        else:
            to_remove.append(index)

mrna_dataset["beta"] = mrna_dataset["beta"].drop(to_remove)
mrna_dataset["pheno"] = mrna_dataset["pheno"].drop(to_remove)

# Classification with ML and DL models
params = {"input_shape": mrna_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.3, "output_shape": len(mrna_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(mrna_dataset,
                                          NeuralClassifier,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

print("Validation accuracy: {} - Test accuracy: {}".format(val_res, test_res))
svm_val, svm_test = benchmark_svm(mrna_dataset, "subtype")
print("SVM validation accuracy: {} - SVM test accuracy: {}".format(svm_val, svm_test))
knn_val, knn_test = benchmark_knn(mrna_dataset, "subtype")
print("KNN validation accuracy: {} - KNN test accuracy: {}".format(knn_val, knn_test))
rf_val, rf_test = benchmark_rf(mrna_dataset, "subtype")
print("RF validation accuracy: {} - RF test accuracy: {}".format(rf_val, rf_test))

with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format(str(mrna_encoder), "MLP", ld, val_res, test_res))
    logfile.write(base.format(str(mrna_encoder), "SVM", ld, svm_val, svm_test))
    logfile.write(base.format(str(mrna_encoder), "KNN", ld, knn_val, knn_test))
    logfile.write(base.format(str(mrna_encoder), "RF", ld, rf_val, rf_test))
