from models import MethylationArrayGenerator
from models.classifiers import NeuralClassifier
import pickle
import pandas as pd

training_set = MethylationArrayGenerator("../data/breast_embedded/train_oe.pkl", "subtype", batch_size=8)
validation_set = MethylationArrayGenerator("../data/breast_embedded/val_oe.pkl", "subtype", batch_size=8)

nn = NeuralClassifier(100, model_serialization_path="../data/models/classifier/")
nn.fit(training_set, validation_set, 200)

test_set = pickle.load(open("../data/breast_embedded/test_oe.pkl", "rb"))
X_test = test_set["beta"].to_numpy()
y_real = pd.get_dummies(test_set["pheno"]["subtype"])
y_pred = nn.predict(X_test)
print(y_pred, y_real)