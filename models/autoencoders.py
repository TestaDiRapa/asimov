from abc import abstractmethod
from models import AbstractModel
from sklearn import preprocessing
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd


class AbstractAutoencoder(AbstractModel):
    """
    An abstract class to define a common interface for all the autoencoders
    """

    def __init__(self,  model_serialization_path="models/autoencoder/", model_name="autoencoder"):
        """
        Class constructor
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.__encoder = None
        self.__decoder = None

    def generate_encoder(self, input_layer, output_layer):
        """
        Instantiates the model that is the encoder part of the network
        :param input_layer: the input layer
        :param output_layer: the output layer
        :return: None
        """
        self.__encoder = Model(inputs=input_layer, outputs=output_layer)

    def encoder_layer(self, encoder_input):
        """
        Abstraction of the encoder as keras layer
        :param encoder_input: the input
        :return: encoder(encoder_input)
        """
        return self.__encoder(encoder_input)

    def generate_decoder(self, input_layer, output_layer):
        """
        Instantiates the model that is the encoder part of the network
        :param input_layer: the input layer
        :param output_layer: the output layer
        :return: None
        """
        self.__decoder = Model(inputs=input_layer, outputs=output_layer)

    def decoder_layer(self, decoder_input):
        """
        Abstraction of the decoder as keras layer
        :param decoder_input: the input
        :return: decoder(decoder_input)
        """
        return self.__decoder(decoder_input)

    @abstractmethod
    def generate_model(self, input_shape):
        """
        Method that the subclasses must implement, defines the model itself
        :param input_shape: the input shape
        :return: None
        """
        pass

    def decode(self, code):
        """
        Decodes a code
        :param code: The code to decode
        :return: the decoded code
        """
        return self.__decoder.predict(code)

    def encode(self, x_input):
        """
        Encodes an input to a code
        :param x_input: the input to encode
        :return: the encoded input
        """
        return self.__encoder.predict(x_input)

    def encode_methylation_array(self, methylation_array):
        """
        This function creates an embedded representation of a methylation array
        :param methylation_array: the methylation array to transform
        :return: another methylation array
        """
        scaler = preprocessing.MinMaxScaler()
        scaled_betas = scaler.fit_transform(methylation_array["beta"].values)
        print(scaled_betas.shape)
        embedded_betas = self.__encoder.predict(scaled_betas)
        print(embedded_betas.shape)
        new_betas = pd.DataFrame(data=embedded_betas, index=list(methylation_array["pheno"].index.values))
        print(new_betas.shape)
        return {"beta": new_betas, "pheno": methylation_array["pheno"]}


class MiRNAEncoder(AbstractAutoencoder):
    """
    An autoencoder for mirna data
    """

    def __init__(self,  input_shape, latent_dimension=100, model_serialization_path="models/autoencoder/", model_name="autoencoder"):
        """
        Class constructor
        :param input_shape: the size of the input
        :param latent_dimension: the size of  the output
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.generate_model(input_shape)
        self.latent_dimension = latent_dimension

    def generate_model(self, input_shape):
        """
        Instantiates encoder, decoder and model as NN with 3 hidden layers
        :param input_shape: the input shape
        :return: None
        """
        hidden_neurons_1 = 1200
        hidden_neurons_2 = 600
        hidden_neurons_3 = 300
        
        encoder_input_layer = Input(shape=input_shape)
        encoder_layer_1 = Dense(hidden_neurons_1, activation="relu")(encoder_input_layer)
        encoder_layer_2 = Dense(hidden_neurons_2, activation="relu")(encoder_layer_1)
        encoder_layer_3 = Dense(hidden_neurons_3, activation="relu")(encoder_layer_2)
        encoder_output = Dense(self.latent_dimension, activation="sigmoid")(encoder_layer_3)
        self.generate_encoder(encoder_input_layer, encoder_output)

        decoder_input_layer = Input(shape=self.latent_dimension)
        decoder_layer_1 = Dense(hidden_neurons_3, activation="relu")(decoder_input_layer)
        decoder_layer_2 = Dense(hidden_neurons_2, activation="relu")(decoder_layer_1)
        decoder_layer_3 = Dense(hidden_neurons_1, activation="relu")(decoder_layer_2)
        decoder_output = Dense(input_shape, activation="sigmoid")(decoder_layer_3)
        self.generate_decoder(decoder_input_layer, decoder_output)

        model_input = Input(shape=input_shape)
        code = self.encoder_layer(model_input)
        model_output = self.decoder_layer(code)
        self.compile_model(model_input, model_output, Adam(lr=0.001), CategoricalCrossentropy())
