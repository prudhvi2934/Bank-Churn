from churn.pipline.training_pipeline import TrainPipeline
from churn.pipline.prediction_pipeline import ChurnClassifier
from pandas import DataFrame
from churn.exception import ChurnException
from churn.logger import logging
# obj = TrainPipeline()
# obj.run_pipeline()

# 716299608, Attrited Customer, 58, F, 3, Unknown, Divorced, Less than $40K, Blue, 36, 2, 2, 3, 1508.0, 0, 1508.0, 0.458, 1990, 31, 0.192, 0.0
A_input_data = {
    "Customer_Age": [58],
    "Gender": ['F'],
    "Dependent_count": [3],
    "Education_Level": ['Unknown'],
    "Marital_Status": ['Divorced'],
    "Income_Category": ['Less than $40K'],
    "Card_Category": ['Blue'],
    "Months_on_book": [36],
    "Total_Relationship_Count": [2],
    "Months_Inactive_12_mon": [2],
    "Contacts_Count_12_mon": [3],
    "Credit_Limit": [1508.0],
    "Total_Revolving_Bal": [0],
    "Avg_Open_To_Buy": [1508.0],
    "Total_Amt_Chng_Q4_Q1": [0.458],
    "Total_Trans_Amt": [1990],
    "Total_Trans_Ct": [31],
    "Total_Ct_Chng_Q4_Q1": [0.192],
    "Avg_Utilization_Ratio": [0.0],
}
# 804708633, Existing Customer, 44, F, 3, Uneducated, Single, Less than $40K, Blue, 39, 3, 2, 2, 7323.0, 0, 7323.0, 0.518, 4666, 87, 0.776, 0.0
E_input_data = {
    "Customer_Age": [44],
    "Gender": ['F'],
    "Dependent_count": [3],
    "Education_Level": ['Uneducated'],
    "Marital_Status": ['Single'],
    "Income_Category": ['Less than $40K'],
    "Card_Category": ['Blue'],
    "Months_on_book": [39],
    "Total_Relationship_Count": [3],
    "Months_Inactive_12_mon": [2],
    "Contacts_Count_12_mon": [2],
    "Credit_Limit": [7323.0],
    "Total_Revolving_Bal": [0],
    "Avg_Open_To_Buy": [7323.0],
    "Total_Amt_Chng_Q4_Q1": [0.518],
    "Total_Trans_Amt": [4666],
    "Total_Trans_Ct": [87],
    "Total_Ct_Chng_Q4_Q1": [0.776],
    "Avg_Utilization_Ratio": [0.0],
}


df = DataFrame(A_input_data)
obj2 = ChurnClassifier()
obj2.predict(df)
