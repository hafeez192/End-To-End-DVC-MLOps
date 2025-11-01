import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml


# ensure the log directory exist 

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration 

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# add function to get params from params.yaml

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

def load_data(file_path:str) -> pd.DataFrame:
    # this file_path getting data from interim folder in data folder 

    # load data from csv file processed data

    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("Data loaded, NaN filled from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise


def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features: int) -> tuple:
    # apply tfidf to preprocessed data 

    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values

        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug("tfidf applied, data transformed")
        return train_df, test_df
    except Exception as e:
        logger.error("Error during tfidf transform: %s",e)
        raise


def save_data(df: pd.DataFrame, file_path:str) -> None:
    # save tfidf data into file 
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data Saved to: %s", file_path)
    except Exception as e:
        logger.error("Unexpected Error while saving data: %s",e)
        raise

def main():
    try:
        # max_features = 50
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']

        # max_features corresponds to number of columns in feature engineered dataset , 50 columns with zero or 1 value
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df , test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df,os.path.join('./data','processed','train_tfidf.csv'))
        save_data(test_df, os.path.join('./data','processed','test_tfidf.csv'))
    
    except Exception as e:
        logger.error('Failed to commplete Feature Engineering: %s',e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()