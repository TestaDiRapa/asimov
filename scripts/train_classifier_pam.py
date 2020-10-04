from methylnet_utils import merge_methylation_arrays, split_methylation_array_by_pheno
from models import methylation_array_kcv
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier, Daneel, Jander, SmallClassifier, \
    SmallConvolutionalClassifier, MediumClassifier
from models.generators import MethylationArrayGenerator
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import pandas as pd
import pickle


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


logfile_name = "../data/classifiers_pam_stats.tsv"
if not os.path.exists(logfile_name):
    with open(logfile_name, "w") as logfile:
        fields = ["MODEL", "USE_METHYLATION", "USE_MIRNA", "USE_MRNA", "USE_CNV", "DROPOUT_RATE", "VALIDATION_ACCURACY",
                  "TEST_ACCURACY", '\n']
        logfile.write('\t'.join(fields))

# Alias for the model
model = Jander

# Logging parameters
methylation = True
mirna = True
mrna = True
cnv = False
dropout = 0.3
dataset_list = []

# Reading PAM50 genes
pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')

# miRNA related to PAM50 genes
mirnas = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))
pam50_mirnas = list()
for mirna, genes in mirnas.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.7:
            pam50_mirnas.append(mirna)

tmp_genes = list()
for m in pam50_mirnas:
    for gene in mirnas[m]:
        if gene[2] > 0.7:
            tmp_genes.append(gene[0])

# ENSG of PAM50 genes
pam50_ENSG = open("../data/PAM50_ENSG.txt").read().split('\n')

# PAM50 CpG islands
islands = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
pam50_cpg = set()
for gene in pam50_genes:
    if gene in islands:
        pam50_cpg = pam50_cpg.union(islands[gene])

# Opens methylation dataset
if methylation:
    methylation_dataset = pickle.load(open("../data/methylation_embedded_pam.pkl", "rb"))
    dataset_list.append(methylation_dataset)

# Opens miRNA dataset
if mirna:
    mirna_dataset = pickle.load(open("../data/mirna_embedded.pkl", "rb"))
    # mirna_scaler = StandardScaler()
    # mirna_dataset["beta"][mirna_dataset["beta"].columns] = \
    #     mirna_scaler.fit_transform(mirna_dataset["beta"][mirna_dataset["beta"].columns])
    # mirna_dataset["beta"] = mirna_dataset["beta"][pam50_mirnas]
    dataset_list.append(mirna_dataset)

# Opens mRNA dataset
if mrna:
    mrna_dataset = pickle.load(open("../data/mrna_embedded_pam.pkl", "rb"))
    # mrna_scaler = StandardScaler()
    # mrna_dataset["beta"][mrna_dataset["beta"].columns] = \
    #     mrna_scaler.fit_transform(mrna_dataset["beta"][mrna_dataset["beta"].columns])
    # mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])[pam50_ENSG]
    dataset_list.append(mrna_dataset)

# Opens CNV dataset
if cnv:
    cnv_dataset = pickle.load(open("../data/cnv_embedded_pam.pkl", "rb"))
    dataset_list.append(cnv_dataset)

# Merges the arrays
final_dataset = merge_methylation_arrays(*dataset_list)

# Just a check on ground truth
gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
gt_index = list(gt_check.index.values)
to_remove = list()
for index, row in final_dataset["pheno"].iterrows():
    if row["subtype"] != "control":
        barcode = "-".join(index.split("-")[:3])
        if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
            final_dataset["pheno"].at[index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
        else:
            to_remove.append(index)

final_dataset["beta"] = final_dataset["beta"].drop(to_remove)
final_dataset["pheno"] = final_dataset["pheno"].drop(to_remove)

# Removes the controls
# not_controls = final_dataset["pheno"]["subtype"] != "Control"
# final_dataset = {"beta": final_dataset["beta"][not_controls], "pheno": final_dataset["pheno"][not_controls]}
"""
params = {"input_shape": final_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": dropout, "output_shape": len(final_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(final_dataset,
                                          model,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="loss", min_delta=0.05, patience=20)])

print("Validation accuracy: {} - Test accuracy: {}".format(val_res, test_res))
svm_val, svm_test = benchmark_svm(final_dataset, "subtype")
print("SVM validation accuracy: {} - SVM test accuracy: {}".format(svm_val, svm_test))
knn_val, knn_test = benchmark_knn(final_dataset, "subtype")
print("KNN validation accuracy: {} - KNN test accuracy: {}".format(knn_val, knn_test))
rf_val, rf_test = benchmark_rf(final_dataset, "subtype")
print("RF validation accuracy: {} - RF test accuracy: {}".format(rf_val, rf_test))

with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format("PAMClassifier'", methylation, mirna, mrna, cnv, dropout, val_res, test_res))
    logfile.write(base.format("SVM", methylation, mirna, mrna, cnv, dropout, svm_val, svm_test))
    logfile.write(base.format("KNN", methylation, mirna, mrna, cnv, dropout, knn_val, knn_test))
    logfile.write(base.format("RF", methylation, mirna, mrna, cnv, dropout, rf_val, rf_test))
"""
# NeuralClassifier, ConvolutionalClassifier,Daneel, Jander,SmallClassifier,SmallConvolutionalClassifier,MediumClassifier
stats = {
    NeuralClassifier: [],
    ConvolutionalClassifier: [],
    Daneel: [],
    Jander: [],
    SmallClassifier: [],
    SmallConvolutionalClassifier: [],
    MediumClassifier: []
}
for i in range(10):
    training_set, validation_set, test_set = \
        split_methylation_array_by_pheno(final_dataset, "subtype", val_rate=0.1, test_rate=0.1)

    for model in stats.keys():
        _, acc = train_dnn_classifier(model, training_set, validation_set, test_set)
        stats[model].append(acc)


for omic, accuracies in stats.items():
    print(omic, np.mean(accuracies))
