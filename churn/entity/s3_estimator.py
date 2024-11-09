# from churn.cloud_storage.aws_storage import SimpleStorageService
import pickle
import os
from churn.entity.estimator import ChurnaModel
import sys
from pandas import DataFrame
from churn.exception import ChurnException
from churn.logger import logging


class USvisaEstimator:
    """
    This class is used to save and retrieve us_visas model in s3 bucket and to do prediction
    """

    def __init__(self, bucket_name, model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        # self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: ChurnaModel = None
        self.model_location = os.path.join(
            './S3_bucket', self.bucket_name)

    def is_model_present(self, model_path):
        try:
            model_full_path = os.path.join(self.model_location, model_path)
            return os.path.exists(model_full_path)
        except Exception as e:
            raise ChurnException(e, sys)
            return False

    def load_model(self,) -> ChurnaModel:
        """
        Load the model from the model_path
        :return:
        """
        try:
            # Set the local path where the model is stored (in the 'artifacts' folder)
            model_path = os.path.join(
                './S3_bucket', self.bucket_name, self.model_path)

            # Check if the model file exists
            if os.path.exists(model_path):
                with open(model_path, "rb") as model_file:
                    # Load the model using pickle
                    best_model = pickle.load(model_file)

                return best_model  # Return the loaded model
            else:
                logging.info(f"Model not found at path: {model_path}")
                return None
        except Exception as e:
            raise ChurnException(e, sys)

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            destination_path = os.path.join(
                self.model_location, self.model_path)

            # Copy the model file to the target location
            with open(from_file, 'rb') as model_file:
                model_data = model_file.read()
            with open(destination_path, 'wb') as dest_file:
                dest_file.write(model_data)

            # Optionally remove the local model file
            if remove:
                os.remove(from_file)

        except Exception as e:
            raise ChurnException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise ChurnException(e, sys)
