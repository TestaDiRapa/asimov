from dataset import filter_expression_by_rate
from models import methylation_array_kcv
from models.autoencoders import MRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Define the latent dimension
ld = 200

# Load the dataset and filter the mRNA keeping only the ones that are non 0 for a certain rate of samples
dataset = pickle.load(open("../data/mrna_exp.pkl", "rb"))
over_rate_mrna = filter_expression_by_rate(dataset, 0.9)
# dataset = pickle.load(open("../data/mrna_exp_all.pkl", "rb"))[over_rate_mrna]
dataset = dataset[over_rate_mrna]

# Generation of training and validation set
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])

# Autoencoder training
mrna_encoder = MRNAEncoder(dataset.shape[1], latent_dimension=ld, model_serialization_path="../data/models/")
mrna_encoder.fit(training_set, validation_set, 2000,
                 callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])

# Creating an embedded representation of the mRNA methylation array
mrna_to_encode = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_to_encode["beta"] = mrna_to_encode["beta"][over_rate_mrna]
mrna_dataset = mrna_encoder.encode_methylation_array(mrna_to_encode)
pickle.dump(mrna_dataset, open("../data/mrna_embedded.pkl", "wb"))

# Classification with ML and DL models
params = {"input_shape": mrna_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.3, "output_shape": len(mrna_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(mrna_dataset,
                                          NeuralClassifier,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
print("Validation accuracy: {}\nTest accuracy: {}".format(val_res, test_res))
print("SVM accuracy: {}".format(benchmark_svm(mrna_dataset, "subtype", verbose=0)))
print("KNN accuracy: {}".format(benchmark_knn(mrna_dataset, "subtype", verbose=0)))
print("RF accuracy: {}".format(benchmark_rf(mrna_dataset, "subtype", verbose=0)))
