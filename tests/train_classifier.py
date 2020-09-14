from models import MethylationArrayGenerator
from models.classifiers import NeuralClassifier

training_set = MethylationArrayGenerator("../data/breast_embedded/train_oe.pkl", "subtype", batch_size=8)
validation_set = MethylationArrayGenerator("../data/breast_embedded/val_oe.pkl", "subtype", batch_size=8)

nn = NeuralClassifier(100, model_serialization_path="../data/models/classifier/")
nn.fit(training_set, validation_set, 200)
