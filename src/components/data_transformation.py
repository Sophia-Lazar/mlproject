import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifscts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def data_tranformation_objet(self):
        ''' 
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']
            
            num_pipeline = ColumnTransformer(
                steps=[
                    ('imputer',SimpleImputer(statergy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = ColumnTransformer(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_freqent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            preprocessor=CustomException(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data")
        except:
            pass