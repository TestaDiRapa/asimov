from models.autoencoders import MiRNAEncoder
from models.generators import AutoencoderGenerator
import numpy as np
import pickle

dataset = pickle.load(open("../data/mirna_exp.pkl", "rb"))
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])
mirna_encoder = MiRNAEncoder()
mirna_encoder.fit(training_set, validation_set, 2000)
for line in dataset.iloc[:val_size, :].to_numpy():
    el_1 = mirna_encoder.predict(line)
    el_2 = mirna_encoder.decode(mirna_encoder.encode(line))
    print(np.sum(el_1-el_2))
