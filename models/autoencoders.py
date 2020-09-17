from abc import abstractmethod
from models import AbstractModel


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

    @abstractmethod
    def generate_model(self, input_shape):
        """
        Method that the subclasses must implement, defines the model itself
        :param input_shape: the input shape
        :return: None
        """
        pass


class MiRNAAutoencoder(AbstractAutoencoder):
    """
    An autoencoder for mirna data
    """

    def __init__(self,  model_serialization_path="models/autoencoder/", model_name="autoencoder"):
        """
        Class constructor
        :param model_serialization_path: the path where to save the model
        :param model_name: the model name
        """
        super().__init__(model_serialization_path, model_name)