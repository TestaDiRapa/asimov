from models import AbstractModel
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class NeuralClassifier(AbstractModel):
    """
    Implements the abstract model as a fully connected NN with 1 hidden layers
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        pre = Dropout(self.dropout_rate)(input_layer)
        hl_1 = Dense(4000, activation="relu")(pre)
        hl_d = Dropout(self.dropout_rate)(hl_1)
        output_layer = Dense(output_shape, activation="softmax")(hl_d)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "NeuralClassifier"


class ConvolutionalClassifier(AbstractModel):
    """
    Implements the abstract model as a convolutional neural network
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        reshaped_input = Reshape((input_shape, 1))(input_layer)
        drop_0 = Dropout(self.dropout_rate)(reshaped_input)
        conv_1 = Conv1D(32, 2, activation="relu")(drop_0)
        conv_2 = Conv1D(64, 2, activation="relu")(conv_1)
        drop_1 = Dropout(self.dropout_rate)(conv_2)
        conv_3 = Conv1D(32, 2, activation="relu")(drop_1)
        conv_4 = Conv1D(16, 2, activation="relu")(conv_3)
        conv_5 = Conv1D(8, 2, activation="relu")(conv_4)
        conv_6 = Conv1D(4, 2, activation="relu")(conv_5)
        flat = Flatten()(conv_6)
        drop_2 = Dropout(self.dropout_rate)(flat)
        output_layer = Dense(output_shape, activation="softmax")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "ConvolutionalClassifier"


class Daneel(AbstractModel):
    """
    Implements the abstract model as a fully connected NN with 3 hidden layers
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        relu_1 = Dense(64, activation="relu")(input_layer)
        drop_1 = Dropout(self.dropout_rate)(relu_1)
        relu_2 = Dense(64, activation="relu")(drop_1)
        drop_2 = Dropout(self.dropout_rate)(relu_2)
        output_layer = Dense(output_shape, activation="softmax")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "Daneel"


class Jander(AbstractModel):
    """
    Implements the abstract model as a fully connected NN with 3 hidden layers
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        relu_1 = Dense(64, activation="relu")(input_layer)
        relu_2 = Dense(64, activation="relu")(relu_1)
        drop_1 = Dropout(self.dropout_rate)(relu_2)
        relu_3 = Dense(64, activation="relu")(drop_1)
        relu_4 = Dense(64, activation="relu")(relu_3)
        drop_2 = Dropout(self.dropout_rate)(relu_4)
        output_layer = Dense(output_shape, activation="softmax")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "Jander"


class SmallClassifier(AbstractModel):
    """
    Implements the abstract model as a fully connected NN with 3 hidden layers
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        relu_1 = Dense(64, activation="relu")(input_layer)
        drop_2 = Dropout(self.dropout_rate)(relu_1)
        output_layer = Dense(output_shape, activation="softmax")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "SmallClassifier"


class MediumClassifier(AbstractModel):
    """
    Implements the abstract model as a fully connected NN with 3 hidden layers
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        relu_1 = Dense(256, activation="relu")(input_layer)
        relu_2 = Dense(256, activation="relu")(relu_1)
        drop_2 = Dropout(self.dropout_rate)(relu_2)
        output_layer = Dense(output_shape, activation="softmax")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "MediumClassifier"


class SmallConvolutionalClassifier(AbstractModel):
    """
    Implements the abstract model as a convolutional neural network
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.2, model_serialization_path="models/classifier/",
                 model_name="classifier"):
        """
        Class constructor
        :param input_shape: the input shape as int
        :param output_shape: the output shape as int
        :param dropout_rate: the dropout rate
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)
        self.dropout_rate = dropout_rate
        self.generate_model(input_shape, output_shape)

    def generate_model(self, input_shape, output_shape):
        """
        Generates the model
        :param input_shape: the input shape
        :param output_shape: the output shape
        :return: None
        """
        input_layer = Input(shape=input_shape)
        reshaped_input = Reshape((input_shape, 1))(input_layer)
        conv_1 = Conv1D(32, 8, activation="relu")(reshaped_input)
        drop_1 = Dropout(self.dropout_rate)(conv_1)
        conv_6 = Conv1D(4, 2, activation="relu")(drop_1)
        flat = Flatten()(conv_6)
        drop_2 = Dropout(self.dropout_rate)(flat)
        output_layer = Dense(output_shape, activation="softmax")(drop_2)

        self.compile_model(input_layer, output_layer, Adam(lr=0.001), CategoricalCrossentropy())

    def __str__(self):
        """
        :return: string representation
        """
        return "SmallConvolutionalClassifier"
