from dataset import filter_expression_by_rate
from methylnet_utils import merge_methylation_arrays
from models import methylation_array_kcv
from models.autoencoders import MRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from models.generators import AutoencoderGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle

ld = 100
dataset = pickle.load(open("../data/mrna_exp.pkl", "rb"))
over_rate_mrna = filter_expression_by_rate(dataset, 0.6)
dataset = dataset[over_rate_mrna]
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])
mrna_encoder = MRNAEncoder(dataset.shape[1], latent_dimension=ld, model_serialization_path="../data/models/")
mrna_encoder.fit(training_set, validation_set, 2000,
                 callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
mrna_to_encode = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_to_encode["beta"] = mrna_to_encode["beta"][over_rate_mrna]
mirna_dataset = mrna_encoder.encode_methylation_array(mrna_to_encode)
final_dataset = mrna_encoder
# methylation_dataset = pickle.load(open("../data/breast_embedded/data_embedded.pkl", "rb"))
# final_dataset = merge_methylation_arrays(mirna_dataset, methylation_dataset)
# not_controls = final_dataset["pheno"]["subtype"] != "Control"
# final_dataset = {"beta": final_dataset["beta"][not_controls], "pheno": final_dataset["pheno"][not_controls]}
params = {"input_shape": final_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.3,"output_shape": len(final_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(final_dataset,
                                          ConvolutionalClassifier,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
print("Validation accuracy: {}\nTest accuracy: {}".format(val_res, test_res))
print("SVM accuracy: {}".format(benchmark_svm(final_dataset, "subtype")))
print("KNN accuracy: {}".format(benchmark_knn(final_dataset, "subtype")))
print("RF accuracy: {}".format(benchmark_rf(final_dataset, "subtype")))