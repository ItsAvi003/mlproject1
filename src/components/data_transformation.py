import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_obj(self):
        '''
        This is resposible for Data Transformation
        '''
        try:
            num_features=['reading_score', 'writing_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']


            #Creating Pipelines for num and cat
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Numerial and Categorical Column is Completed")

            preprocessor=ColumnTransformer(
                [
                    ("cat_pipelines",cat_pipeline,cat_features),
                    ("num_pipelines",num_pipeline,num_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys) 


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)    
            logging.info("Read Train and Test data complete")


            logging.info("obtaining preprocessor object")
            preprpcessor_obj=self.get_data_transformation_obj()

            target_column_name="math_score"
            num_features=['math_score', 'reading_score', 'writing_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            input_feature_train_df=train_df.drop(columns=["math_score"],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=["math_score"],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying prerprocession on train and test data")
            
            #Applying preprocessiong on train and test
            input_feature_train_arr=preprpcessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprpcessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing obj")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprpcessor_obj
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)
