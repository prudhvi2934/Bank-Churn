import os
import sys

import numpy as np
import pandas as pd
from churn.entity.config_entity import ChurnPredictorConfig
from churn.entity.s3_estimator import USvisaEstimator

from churn.utils.main_utils import read_yaml_file
from pandas import DataFrame
from churn.exception import ChurnException
from churn.logger import logging


class ChurnData:
    def __init__(self,
                 Customer_Age,
                 Gender,
                 Dependent_count,
                 Education_Level,
                 Marital_Status,
                 Income_Category,
                 Card_Category,
                 Months_on_book,
                 Total_Relationship_Count,
                 Months_Inactive_12_mon,
                 Contacts_Count_12_mon,
                 Credit_Limit,
                 Total_Revolving_Bal,
                 Avg_Open_To_Buy,
                 Total_Amt_Chng_Q4_Q1,
                 Total_Trans_Amt,
                 Total_Trans_Ct,
                 Total_Ct_Chng_Q4_Q1,
                 Avg_Utilization_Ratio
                 ):
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Customer_Age = Customer_Age
            self.Gender = Gender
            self.Dependent_count = Dependent_count
            self.Education_Level = Education_Level
            self.Marital_Status = Marital_Status
            self.Income_Category = Income_Category
            self.Card_Category = Card_Category
            self.Months_on_book = Months_on_book
            self.Total_Relationship_Count = Total_Relationship_Count
            self.Months_Inactive_12_mon = Months_Inactive_12_mon
            self.Contacts_Count_12_mon = Contacts_Count_12_mon
            self.Credit_Limit = Credit_Limit
            self.Total_Revolving_Bal = Total_Revolving_Bal
            self.Avg_Open_To_Buy = Avg_Open_To_Buy
            self.Total_Amt_Chng_Q4_Q1 = Total_Amt_Chng_Q4_Q1
            self.Total_Trans_Amt = Total_Trans_Amt
            self.Total_Trans_Ct = Total_Trans_Ct
            self.Total_Ct_Chng_Q4_Q1 = Total_Ct_Chng_Q4_Q1
            self.Avg_Utilization_Ratio = Avg_Utilization_Ratio

        except Exception as e:
            print(e, sys)

    def get_churn_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from churn class input
        """
        try:

            churn_input_dict = self.get_churn_data_as_dict()
            return DataFrame(churn_input_dict)

        except Exception as e:
            print(e, sys)

    def get_churn_data_as_dict(self):
        """
        This function returns a dictionary from churn class input 
        """
        print(
            "Entered get_churn_data_as_dict method as churn class")

        try:
            input_data = {
                "Customer_Age": [self.Customer_Age],
                "Gender": [self.Gender],
                "Dependent_count": [self.Dependent_count],
                "Education_Level": [self.Education_Level],
                "Marital_Status": [self.Marital_Status],
                "Income_Category": [self.Income_Category],
                "Card_Category": [self.Card_Category],
                "Months_on_book": [self.Months_on_book],
                "Total_Relationship_Count": [self.Total_Relationship_Count],
                "Months_Inactive_12_mon": [self.Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [self.Contacts_Count_12_mon],
                "Credit_Limit": [self.Credit_Limit],
                "Total_Revolving_Bal": [self.Total_Revolving_Bal],
                "Avg_Open_To_Buy": [self.Avg_Open_To_Buy],
                "Total_Amt_Chng_Q4_Q1": [self.Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [self.Total_Trans_Amt],
                "Total_Trans_Ct": [self.Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [self.Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [self.Avg_Utilization_Ratio],
            }

            print("Created churn data dict")

            print("Exited get_churn_data_as_dict method as USvisaData class")

            return input_data

        except Exception as e:
            print(e, sys)


class ChurnClassifier:
    def __init__(self, prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            print(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            print("Entered predict method of USvisaClassifier class")
            model = USvisaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            print(f"****************{result}***********************")
            return result

        except Exception as e:
            print(e, sys)
