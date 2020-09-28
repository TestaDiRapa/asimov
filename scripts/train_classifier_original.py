from dataset.mirna_exp import get_interactions_over_threshold
from methylnet_utils import merge_methylation_arrays
from models import methylation_array_kcv
from models.classifiers import NeuralClassifier, ConvolutionalClassifier, MOLIClassifier
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle

logfile_name = "../data/classifiers_original_stats.tsv"
if not os.path.exists(logfile_name):
    with open(logfile_name, "w") as logfile:
        fields = ["MODEL", "USE_METHYLATION", "USE_MIRNA", "USE_MRNA", "USE_CNV", "DROPOUT_RATE", "VALIDATION_ACCURACY",
                  "TEST_ACCURACY", '\n']
        logfile.write('\t'.join(fields))

# Alias for the model
model = NeuralClassifier

# Logging parameters
methylation = False
mirna = True
mrna = True
cnv = False
dropout = 0.1
dataset_list = []

# Opens methylation dataset
if methylation:
    methylation_dataset = pickle.load(open("../data/breast_embedded/data_embedded.pkl", "rb"))
    dataset_list.append(methylation_dataset)

# Opens mirna dataset
if mirna:
    mirna_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
    dataset_list.append(mirna_dataset)

# Opens mirna dataset
if mrna:
    over_rate_mrna = get_interactions_over_threshold(0.9, False)
    mrna_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
    mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])[over_rate_mrna]
    dataset_list.append(mrna_dataset)

# Merges the arrays
final_dataset = merge_methylation_arrays(*dataset_list)

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

with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format("NeuralClassifier", methylation, mirna, mrna, cnv, dropout, val_res, test_res))
