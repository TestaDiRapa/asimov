from dataset.mirna_exp import get_interactions_over_threshold
from models import methylation_array_kcv
from models.autoencoders import MiRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

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
over_rate_genes = get_interactions_over_threshold(0.9, True)
genes_cpg = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))

final_cpg = set()
for gene in over_rate_genes:
    if gene in genes_cpg:
        final_cpg = final_cpg.union(genes_cpg[gene])
index = set(dataset["beta"].columns.values)
final_cpg = final_cpg.intersection(index)
dataset["beta"] = dataset["beta"][final_cpg]


# Generation of training and validation set
val_size = int(dataset["beta"].shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset["beta"].iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset["beta"].iloc[val_size:, :])

# Autoencoder training
methylation_encoder = MiRNAEncoder(dataset["beta"].shape[1], latent_dimension=ld,
                                   model_serialization_path="../data/models/")
methylation_encoder.fit(training_set, validation_set, 500,
                        callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

# Creating an embedded representation of the methylation data

methylation_dataset = methylation_encoder.encode_methylation_array(dataset)
print(methylation_dataset)
pickle.dump(methylation_dataset, open("../data/methylation_embedded.pkl", "wb"))

# Classification with ML and DL models
params = {"input_shape": methylation_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.2, "output_shape": len(methylation_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(methylation_dataset,
                                          NeuralClassifier,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

print("Validation accuracy: {} - Test accuracy: {}".format(val_res, test_res))
svm_val, svm_test = benchmark_svm(methylation_dataset, "subtype")
print("SVM validation accuracy: {} - SVM test accuracy: {}".format(svm_val, svm_test))
knn_val, knn_test = benchmark_knn(methylation_dataset, "subtype")
print("KNN validation accuracy: {} - KNN test accuracy: {}".format(knn_val, knn_test))
rf_val, rf_test = benchmark_rf(methylation_dataset  , "subtype")
print("RF validation accuracy: {} - RF test accuracy: {}".format(rf_val, rf_test))

with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format(str(methylation_encoder), "MLP", ld, val_res, test_res))
    logfile.write(base.format(str(methylation_encoder), "SVM", ld, svm_val, svm_test))
    logfile.write(base.format(str(methylation_encoder), "KNN", ld, knn_val, knn_test))
    logfile.write(base.format(str(methylation_encoder), "RF", ld, rf_val, rf_test))
