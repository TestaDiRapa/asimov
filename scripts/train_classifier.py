from methylnet_utils import merge_methylation_arrays
from models import methylation_array_kcv
from models.benchmark import benchmark_svm, benchmark_rf, benchmark_knn
from models.classifiers import NeuralClassifier, ConvolutionalClassifier
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Opens methylation dataset
methylation_dataset = pickle.load(open("../data/breast_embedded/data_embedded.pkl", "rb"))

# Opens mirna dataset
mirna_dataset = pickle.load(open("../data/mirna_embedded.pkl", "rb"))

# Merges the arrays
final_dataset = merge_methylation_arrays(mirna_dataset, methylation_dataset)
print(final_dataset)

# Removes the controls
# not_controls = final_dataset["pheno"]["subtype"] != "Control"
# final_dataset = {"beta": final_dataset["beta"][not_controls], "pheno": final_dataset["pheno"][not_controls]}

params = {"input_shape": final_dataset["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
          "dropout_rate": 0.1, "output_shape": len(final_dataset["pheno"]["subtype"].unique())}
val_res, test_res = methylation_array_kcv(final_dataset,
                                          NeuralClassifier,
                                          params,
                                          "subtype",
                                          callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])

print("Validation accuracy: {}\nTest accuracy: {}".format(val_res, test_res))
print("SVM accuracy: {}".format(benchmark_svm(final_dataset, "subtype")))
print("KNN accuracy: {}".format(benchmark_knn(final_dataset, "subtype")))
print("RF accuracy: {}".format(benchmark_rf(final_dataset, "subtype")))

