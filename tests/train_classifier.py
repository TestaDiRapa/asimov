from models import MethylationArrayGenerator
from models.classifiers import NeuralClassifier
import pickle
import pandas as pd

training_set = MethylationArrayGenerator("../data/breast_embedded/train_oe.pkl", "subtype", batch_size=32)
validation_set = MethylationArrayGenerator("../data/breast_embedded/val_oe.pkl", "subtype", batch_size=32)

nn = NeuralClassifier(100, model_serialization_path="../data/models/classifier/", dropout_rate=0.3)
nn.fit(training_set, validation_set, 100)

test_set = pickle.load(open("../data/breast_embedded/test_oe.pkl", "rb"))
X_test = test_set["beta"].to_numpy()
y_real = pd.get_dummies(test_set["pheno"]["subtype"])
print(nn.evaluate(X_test, y_real.to_numpy()))
