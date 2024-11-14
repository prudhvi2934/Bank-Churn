from google.cloud import storage
from churn.configuration.gcp_connection import GCPservice
from io import StringIO
from typing import Union, List
import os
import sys
from churn.logger import logging
from churn.exception import ChurnException
from pandas import DataFrame, read_csv
from churn.entity.estimator import ChurnModel
import dill


class GCPbucket:
    def __init__(self, bucket_name):
        # Initialize GCP service and storage client
        gcpservice = GCPservice()
        self.storage_client = gcpservice.storage_client
        self.bucket_name = bucket_name

    def upload_file(self, source_file_name, destination_blob_name):
        """
        Uploads a file to the GCP bucket.

        Parameters:
        source_file_name (str): Local path to the file (e.g., 'model.pickle').
        destination_blob_name (str): Name to give the file in the bucket (e.g., 'folder/model.pickle').
        """
        try:
            # Get the bucket
            bucket = self.storage_client.bucket(self.bucket_name)

            # Create a blob object for the destination path in the bucket
            blob = bucket.blob(destination_blob_name)

            # Upload the file
            blob.upload_from_filename(source_file_name)

            logging.info(
                f"File {source_file_name} uploaded to {destination_blob_name}.")
        except Exception as e:
            raise ChurnException(e, sys)

    def model_exists(self, model_name):
        """
        Checks if a file (model) exists in the GCP bucket.

        Parameters:
        model_name (str): The name/path of the file in the bucket.

        Returns:
        bool: True if the file exists, False otherwise.
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(model_name)
            return blob.exists()
        except Exception as e:
            print(f"An error occurred while checking if model exists: {e}")
            return False

    def load_model(self, model_name: str, bucket_name: str) -> ChurnModel:
        """
        Downloads and loads a model from the GCP bucket as a ChurnModel object.

        Parameters:
        model_name (str): The name/path of the model file in the bucket.
        bucket_name (str): The name of the bucket where the model is stored.

        Returns:
        ChurnModel: An instance of ChurnModel initialized with preprocessing and trained model objects.
        """
        try:
            # Get the bucket and blob (model file) from GCP
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(model_name)

            # Check if the model exists in the bucket
            if not blob.exists():
                raise FileNotFoundError(
                    f"Model file {model_name} not found in bucket {bucket_name}.")

            # Download the model content as bytes
            model_data = blob.download_as_bytes()

            # Deserialize the model data using dill
            usvisa_model = dill.loads(model_data)

            # Ensure the deserialized object is an instance of ChurnModel
            if not isinstance(usvisa_model, ChurnModel):
                raise TypeError(
                    "The loaded object is not of type 'ChurnModel'.")

            logging.info("Model loaded successfully.")
            return usvisa_model

        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            raise ChurnException(e, sys)
