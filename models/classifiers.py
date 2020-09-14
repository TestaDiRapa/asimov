from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os


class NeuralClassifier:

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/", model_name="classifier"):
        self.model = None
        os.makedirs(model_serialization_path)
        self.serialization_path = model_serialization_path
        self.model_name = model_name
        self.generate_model(input_shape, dropout_rate)

    def generate_model(self, input_shape, dropout_rate):
        input_layer = Input(shape=input_shape)
        hl_1 = Dense(200, activation="relu")(input_layer)
        hl_2 = Dense(300, activation="relu")(hl_1)
        hl_3 = Dense(3000, activation="relu")(hl_2)
        hl_d = Dropout(dropout_rate)(hl_3)
        output_layer = Dense(5, activation="sigmoid")(hl_d)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.save(os.path.join(self.serialization_path, self.model_name+".model"))

        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer,
                           loss="binary_crossentropy",
                           metrics=["accuracy", "loss"])

        self.model.summary()

    def fit(self, training_set, validation_set, epochs):
        self.model.fit(
            training_set,
            # batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='loss', min_delta=0.1, patience=10)],
            validation_data=validation_set
        )
        self.model.save_weights(os.path.join(self.serialization_path, self.model_name+".weights"))
