import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs") #--> This ensure my all logs will be separately stored in log file inside my project
os.makedirs(log_path,exist_ok=True)#for not giving error if my log file already exist
LOG_FILE_PATH=os.path.join(log_path,LOG_FILE) # This will be the full path where massage will be written for ex: C:\mlproject1\logs\08_26_2025_11_45_30.log

# Configuration of logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    level=logging.INFO,
)

