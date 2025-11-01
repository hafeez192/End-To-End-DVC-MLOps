import os 
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# ensure log dir exist 

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration 

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str) -> pd.DataFrame:
    # load data as tfidf 
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s", file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s",e)
        raise
    except FileNotFoundError as e:
        logger.error('File not Found: %s',e)
        raise
    except Exception as e:
        logger.error("Unexpected Error occured while loading data: %s",e)
        raise

def train_model(X_train: np.ndarray,y_train: np.ndarray, params:dict) -> RandomForestClassifier:
    # Train RandomForestClassifier 
    # params: dictionary of hyperparameters 
    # return: Trained RandomForestClassifier 

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Samples: X_train, y_train must be same")
        logger.debug("RandomForestClassifer: Initialize with params: %s",params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug("Model Training started with %d samples",X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("Model Training Completed")

        return clf
    except ValueError as e:
        logger.error("Value error during training: %s",e)
        raise
    except Exception as e:
        logger.error("Error during Training: %s",e)
        raise

def save_model(model,file_path:str) ->None:
    # save model to file 

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model,file)
        logger.debug("Model saved: %s",file_path)
    except FileNotFoundError as e:
        logger.error("File path not found: %s",e)
        raise
    except Exception as e:
        logger.error("Error occured while saving model: %s",e)
        raise

def main():

    try:
        params = {'n_estimators': 25, 'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train, y_train,params)

        model_save_path = 'models/model.pkl'
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error("Failed to complete the model building process: %s",e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

