from models import AbstractModel
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class NeuralClassifier(AbstractModel):
    """
    Implements the abstract model as a fully connected NN with 3 hidden layers
    """

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape)

    def generate_model(self, input_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        hl_1 = Dense(200, activation="relu")(input_layer)
        hl_2 = Dense(300, activation="relu")(hl_1)
        hl_3 = Dense(3000, activation="relu")(hl_2)
        hl_d = Dropout(self.dropout_rate)(hl_3)
        output_layer = Dense(6, activation="sigmoid")(hl_d)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())


class ConvolutionalClassifier(AbstractModel):
    """
    Implements the abstract model as a convolutional neural network
    """

    def __init__(self, input_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape)

    def generate_model(self, input_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        reshaped_input = Reshape((input_shape, 1))(input_layer)
        conv_1 = Conv1D(128, 4, activation="relu")(reshaped_input)
        conv_2 = Conv1D(64, 2, activation="relu")(conv_1)
        drop_l = Dropout(self.dropout_rate)(conv_2)
        conv_3 = Conv1D(64, 2, activation="relu")(drop_l)
        flat = Flatten()(conv_3)
        drop_2 = Dropout(self.dropout_rate)(flat)
        output_layer = Dense(6, activation="sigmoid")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())
