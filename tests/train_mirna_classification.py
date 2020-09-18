from methylnet_utils import merge_methylation_arrays
from models import methylation_array_kcv
from models.autoencoders import MiRNAEncoder
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from models.generators import AutoencoderGenerator
import pickle

ld = 100

dataset = pickle.load(open("../data/mirna_exp.pkl", "rb"))
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])
mirna_encoder = MiRNAEncoder(1881, latent_dimension=ld, model_serialization_path="../data/models/")
mirna_encoder.fit(training_set, validation_set, 2000)

mirna_dataset = mirna_encoder.encode_methylation_array(pickle.load(open("../data/mirna_exp_ma.pkl", "rb")))
methylation_dataset = pickle.load(open("../data/breast_embedded/data_embedded.pkl", "rb"))
final_dataset = merge_methylation_arrays(mirna_dataset, methylation_dataset)
params = {"input_shape": ld+100, "model_serialization_path": "../data/models/classifier/", "dropout_rate": 0.3}
val_res, test_res = methylation_array_kcv(final_dataset,
                                          ConvolutionalClassifier,
                                          params,
                                          "subtype")
print("Validation accuracy: {}\nTest accuracy: {}".format(val_res, test_res))
print("SVM accuracy: {}".format(benchmark_svm(final_dataset, "subtype")))
print("KNN accuracy: {}".format(benchmark_knn(final_dataset, "subtype")))
print("RF accuracy: {}".format(benchmark_rf(final_dataset, "subtype")))
