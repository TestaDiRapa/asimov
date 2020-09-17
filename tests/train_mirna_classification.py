from models import methylation_array_kcv
from models.autoencoders import MiRNAEncoder
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from models.generators import AutoencoderGenerator
import pickle


dataset = pickle.load(open("../data/mirna_exp.pkl", "rb"))
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])
ld = 50
mirna_encoder = MiRNAEncoder(1881, latent_dimension=ld, model_serialization_path="../data/models/")
mirna_encoder.fit(training_set, validation_set, 2000)
class_dataset = mirna_encoder.encode_methylation_array(pickle.load(open("../data/mirna_exp_ma.pkl", "rb")))
params = {"input_shape": ld, "model_serialization_path": "../data/models/classifier/", "dropout_rate": 0.3}
val_res, test_res = methylation_array_kcv(class_dataset,
                                          NeuralClassifier,
                                          params,
                                          "subtype")
print("Validation accuracy: {}\nTest accuracy: {}".format(val_res, test_res))
