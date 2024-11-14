from churn.cloud_storage.gcp_storage import GCPbucket
import pickle
import os
from churn.entity.estimator import ChurnModel
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
        self.gcp = GCPbucket(self.bucket_name)
        self.model_path = model_path
        self.loaded_model: ChurnModel = None
        # self.model_location = os.path.join(
        #     './S3_bucket', self.bucket_name)

    def is_model_present(self, model_path):
        try:
            return self.gcp.model_exists(model_name=model_path)
        except ChurnException as e:
            print(e)
            return False

    def load_model(self,) -> ChurnModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.gcp.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.gcp.upload_file(from_file,
                                 destination_blob_name=self.model_path)
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
