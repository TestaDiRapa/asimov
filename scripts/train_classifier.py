from methylnet_utils import merge_methylation_arrays
from models import methylation_array_kcv
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier, MOLIClassifier
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
import pickle

logfile_name = "../data/classifiers_stats.tsv"
if not os.path.exists(logfile_name):
    with open(logfile_name, "w") as logfile:
        fields = ["MODEL", "USE_METHYLATION", "USE_MIRNA", "USE_MRNA", "USE_CNV", "DROPOUT_RATE", "VALIDATION_ACCURACY",
                  "TEST_ACCURACY", '\n']
        logfile.write('\t'.join(fields))

# Alias for the model
model = MOLIClassifier

# Logging parameters
methylation = True
mirna = True
mrna = True
cnv = False
dropout = 0.3
dataset_list = []

# Opens methylation dataset
if methylation:
    methylation_dataset = pickle.load(open("../data/methylation_embedded.pkl", "rb"))
    dataset_list.append(methylation_dataset)

# Opens mirna dataset
if mirna:
    mirna_dataset = pickle.load(open("../data/mirna_embedded.pkl", "rb"))
    dataset_list.append(mirna_dataset)

# Opens mirna dataset
if mrna:
    mrna_dataset = pickle.load(open("../data/mrna_embedded.pkl", "rb"))
    dataset_list.append(mrna_dataset)

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

params = {"input_shape": final_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": dropout, "output_shape": len(final_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(final_dataset,
                                          model,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])

print("Validation accuracy: {} - Test accuracy: {}".format(val_res, test_res))
svm_val, svm_test = benchmark_svm(final_dataset, "subtype")
print("SVM validation accuracy: {} - SVM test accuracy: {}".format(svm_val, svm_test))
knn_val, knn_test = benchmark_knn(final_dataset, "subtype")
print("KNN validation accuracy: {} - KNN test accuracy: {}".format(knn_val, knn_test))
rf_val, rf_test = benchmark_rf(final_dataset, "subtype")
print("RF validation accuracy: {} - RF test accuracy: {}".format(rf_val, rf_test))

with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format("MOLIClassifier'", methylation, mirna, mrna, cnv, dropout, val_res, test_res))
    logfile.write(base.format("SVM", methylation, mirna, mrna, cnv, dropout, svm_val, svm_test))
    logfile.write(base.format("KNN", methylation, mirna, mrna, cnv, dropout, knn_val, knn_test))
    logfile.write(base.format("RF", methylation, mirna, mrna, cnv, dropout, rf_val, rf_test))
