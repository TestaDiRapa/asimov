from models.autoencoders import MiRNAEncoder
from models.generators import AutoencoderGenerator
import numpy as np
import pickle
from sklearn import preprocessing


dataset = pickle.load(open("../data/mirna_exp.pkl", "rb"))
mms = preprocessing.MinMaxScaler()
x = mms.fit_transform(dataset.values)
val_size = int(dataset.shape[0]*0.1)
validation_set = AutoencoderGenerator(dataset.iloc[:val_size, :])
training_set = AutoencoderGenerator(dataset.iloc[val_size:, :])
mirna_encoder = MiRNAEncoder(1881)
mirna_encoder.fit(training_set, validation_set, 2000)
for v in x:
    test_i = np.empty((1, 1881))
    test_i[0] = v
    el_1 = mirna_encoder.predict(test_i)
    el_2 = mirna_encoder.decode(mirna_encoder.encode(test_i))
    print(np.sum(el_1-el_2))
