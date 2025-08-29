import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train_data.csv")
    test_data_path=os.path.join('artifacts',"test_data.csv")
    raw_data_path=os.path.join('artifacts',"raw_data.csv")

class IngetionData:
    def __init__(self):
        self.ingetion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingetion methord or components")

        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Exported the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingetion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingetion_config.raw_data_path,index=False,header=True)
            logging.info("Train Test Split ocuuring")

            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)

            train_set.to_csv(self.ingetion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingetion_config.test_data_path,index=False,header=True)
            logging.info("Ingetion completed")

            return(
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=IngetionData()
    train_dt,test_dt=obj.initiate_data_ingestion()
     
    data_tranformation=DataTransformation()
    # we are just doing the preprocessiong part on train and test
    data_tranformation.initiate_data_transformation(train_dt,test_dt)



