import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml #experiment tracking
from dvclive import Live #experiment tracking

# ensure log exist 

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration 

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    # load params from yaml file , params.yaml
    try:

        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug("parameters retrieved from : %s",params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s",e)
        raise

def load_model(file_path:str):
    # load the pickle file 
    try:

        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("model loaded from : %s", file_path)
        return model
    except FileNotFoundError as e:
        logger.error("File not Found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error:while loading file: %s",e)
        raise


# test data load now ; tfidf 

def load_data(file_path:str) -> pd.DataFrame:
    # load test data from csv 
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse csv file: %s',e)
        raise
    except Exception as e:
        logger.error("Unexpected Error: Load file file: %s",e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    # evaluate and return the metrics
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error('Error during evaluaton: %s',e)
        raise


def save_metrics(metrics: dict, file_path:str) -> None:
    # save evaluaton metrics to json file 

    try:
        # ensure directory exist 
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('metrics saved to: %s', file_path)

    except Exception as e:
        logger.error('Error durind saving metrics: %s',e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(clf,X_test,y_test)

        # experiment tracking using dvc live 

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy',accuracy_score(y_test, y_test))
            live.log_metric('precision',precision_score(y_test,y_test))
            live.log_metric('recall',recall_score(y_test,y_test))

            live.log_params(params)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete model evaludation: %s',e)
        print(f"Error: {e}")

if __name__== '__main__':
    main()