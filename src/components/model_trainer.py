import sys 
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:    
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
           logging.info("Spliting train and test input data")

           X_train,y_train,X_test,y_test=(
               train_array[:,:-1],
               train_array[:,-1],
               test_array[:,:-1],
               test_array[:,-1]
           )
           models={
               
                "LinerRegressor":LinearRegression(),  
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }
           
           params={
               "LinerRegressor":{},
               "K-Neighbors Regressor":{
                   'n_neighbors':[5,7,9,11],
        
               },
               "Decision Tree":{
                   'criterion':['squared_error','friedman_mse','absolute_error','poisson']
               },
               "Random Forest Regressor":{
                   'n_estimators':[8,16,10,32]
               },
               "Gradient Boosting":{
                   'learning_rate':[.1,.01,.05,.001],
                   'subsample':[0.6,0.7,0.75,0.8],
                   'n_estimators':[8,16,10,32]
               },
               "XGBRegressor":{
                   'learning_rate':[.1,.01,.05,.001],
                   'n_estimators':[8,16,10,32]
               },
               "CatBoosting Regressor":{
                   'depth':[6,8,10],
                   'learning_rate':[.1,.01,.05,.001]
                   
               },
               "AdaBoost Regressor":{
                   'learning_rate':[.1,.01,.05,.001],
                   'n_estimators':[8,16,10,32]
               }

           }
           
           model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

           # To get best model score:
           best_model_score=max(sorted(model_report.values()))

           # To get best model name from dict:
           best_model_name=list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
           ]
           best_model=models[best_model_name]

           if best_model_score<0.6:
               raise CustomException("No best model found")
           logging.info("Best model found with best score on training and test data set")

           save_object(
               file_path=self.model_trainer_config.trained_model_file_path,
               obj=best_model
           )
           predicted=best_model.predict(X_test)

           r2_square=r2_score(y_test,predicted)
           return r2_square

        except Exception as e:
            raise CustomException(e,sys)