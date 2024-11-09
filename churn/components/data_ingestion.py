import os
import sys

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from churn.entity.artifact_entity import DataIngestionArtifact
from churn.entity.config_entity import DataIngestionConfig
from churn.logger import logging
from churn.exception import ChurnException


class DataIngestion:

    def __init__(self, dataIngestionConfig: DataIngestionConfig = DataIngestionConfig()):

        try:
            self.dataIngestionConfig = dataIngestionConfig
        except Exception as e:
            raise ChurnException(e, sys)

    def export_data_into_df(self) -> DataFrame:

        try:
            dataframe = pd.read_csv(
                "/Users/prudhvisajja/Documents/Data Science MSc/COP528 AI and ML/CourseWork/BankChurners.csv")
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.dataIngestionConfig.feature_store_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            print("Exception in Data Ingestion class export_data_into_df", e)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:

        try:
            train_data, test_data = train_test_split(
                dataframe, test_size=0.2)

            dir_path = os.path.dirname(
                self.dataIngestionConfig.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_data.to_csv(
                self.dataIngestionConfig.training_file_path, index=False, header=True)
            test_data.to_csv(
                self.dataIngestionConfig.testing_file_path, index=False, header=True)
            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise ChurnException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            dataframe = self.export_data_into_df()
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.dataIngestionConfig.training_file_path, test_file_path=self.dataIngestionConfig.testing_file_path)

            return data_ingestion_artifact
        except Exception as e:
            raise ChurnException(e, sys) from e
