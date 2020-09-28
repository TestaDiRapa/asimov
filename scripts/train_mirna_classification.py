from dataset import filter_expression_by_rate
from models import methylation_array_kcv
from models.autoencoders import MiRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier, MOLIClassifier
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

logfile_name = "../data/mirna_embedding.tsv"
if not os.path.exists(logfile_name):
    with open(logfile_name, "w") as logfile:
        fields = ["MODEL", "ACCURACY_EVALUATION_METHOD", "LATENT_DIMENSION", "VALIDATION_ACCURACY", "TEST_ACCURACY",
                  '\n']
        logfile.write('\t'.join(fields))

# Define the latent dimension
ld = 200

# Load the dataset and filter the mRNA keeping only the ones that are non 0 for a certain rate of samples
dataset = pickle.load(open("../data/mirna_exp.pkl", "rb"))
over_rate_mirna = filter_expression_by_rate(dataset, 0.5)
# dataset = dataset[over_rate_mirna]

# Generation of training and validation set
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])

# Autoencoder training
mirna_encoder = MiRNAEncoder(dataset.shape[1], latent_dimension=ld, model_serialization_path="../data/models/")
mirna_encoder.fit(training_set, validation_set, 500,
                  callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

# Creating an embedded representation of the mRNA methylation array
mirna_to_encode = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_to_encode["beta"] = mirna_to_encode["beta"]  # [over_rate_mirna]
mirna_dataset = mirna_encoder.encode_methylation_array(mirna_to_encode)
pickle.dump(mirna_dataset, open("../data/mirna_embedded.pkl", "wb"))

# Classification with ML and DL models
params = {"input_shape": mirna_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.05, "output_shape": len(mirna_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(mirna_dataset,
                                          MOLIClassifier,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

print("Validation accuracy: {} - Test accuracy: {}".format(val_res, test_res))
svm_val, svm_test = benchmark_svm(mirna_dataset, "subtype")
print("SVM validation accuracy: {} - SVM test accuracy: {}".format(svm_val, svm_test))
knn_val, knn_test = benchmark_knn(mirna_dataset, "subtype")
print("KNN validation accuracy: {} - KNN test accuracy: {}".format(knn_val, knn_test))
rf_val, rf_test = benchmark_rf(mirna_dataset, "subtype")
print("RF validation accuracy: {} - RF test accuracy: {}".format(rf_val, rf_test))

with open(logfile_name, 'a') as logfile:
    base = "{}\t{}\t{}\t{}\t{}\n"
    logfile.write(base.format(str(mirna_encoder), "MLP", ld, val_res, test_res))
    logfile.write(base.format(str(mirna_encoder), "SVM", ld, svm_val, svm_test))
    logfile.write(base.format(str(mirna_encoder), "KNN", ld, knn_val, knn_test))
    logfile.write(base.format(str(mirna_encoder), "RF", ld, rf_val, rf_test))
