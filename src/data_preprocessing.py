import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from category_encoders import CountEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object   

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            numerical_features =  ['C14', 'C17', 'C20']
            
            categorical_features =[
                                    'site_id',
                                    'site_domain',
                                    'site_category',
                                    'app_id',
                                    'app_domain',
                                    'app_category',
                                    'device_id',
                                    'device_ip',
                                    'device_model',
                                    'C1',
                                    'banner_pos',
                                    'device_type',
                                    'device_conn_type',
                                    'C15',
                                    'C16',
                                    'C18',
                                    'C19',
                                    'C21']
            
            low_cardinal_cat_cols = ['C1', 
                                     'banner_pos', 
                                     'device_type', 
                                     'device_conn_type', 
                                     'C15', 
                                     'C16', 
                                     'C18']
            
            med_cardinal_cat_cols = ['site_category', 
                                     'app_domain', 
                                     'app_category', 
                                     'C19', 
                                     'C21']
            
            high_cardinal_cat_cols = ['site_id', 
                                      'site_domain', 
                                      'app_id', 
                                      'device_id', 
                                      'device_ip', 
                                      'device_model']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    #('scaler', MinMaxScaler())
                ]
            )
            low_card_cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                    #('scaler', MinMaxScaler())
                ]
            )
            med_card_cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OrdinalEncoder()),
                    #('scaler', MinMaxScaler())
                ]
            )
            high_card_cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', CountEncoder(normalize=False, handle_unknown=0)),
                    #('scaler', MinMaxScaler())
                ]
            )
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('low_card_cat_pipeline', low_card_cat_pipeline, low_cardinal_cat_cols),
                    ('med_card_cat_pipeline', med_card_cat_pipeline, med_cardinal_cat_cols),
                    ('high_card_cat_pipeline', high_card_cat_pipeline, high_cardinal_cat_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'click'
            drop_columns = [target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            target_feature_train_arr = target_feature_train_df.values
            target_feature_test_arr = target_feature_test_df.values
   
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            logging.info("Preprocessing completed")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)