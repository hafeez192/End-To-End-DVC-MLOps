import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem import PorterStemmer


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


# ensure log directory exist 

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# setting up logger 
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    # preprocess text, lower case, tokenize, stopwords remove etc 
    ps = PorterStemmer()
    # convert to lowercase 
    text = text.lower()
    # tokenize the text 
    text = nltk.word_tokenize(text)
    # remove alphanumeric 
    text = [word for word in text if word.isalnum()]
    # remove stopwords and punct 
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # stem the words 
    text = [ps.stem(word) for word in text]
    # join tokens back into single string 
    return ' '.join(text)

def preprocess_df(df, text_column='text',target_column='target'):
    # preprocess: encode target col, remove duplicate, transform text column 

    try:
        logger.debug("Starting processing for DataFrame")
        # encode target col 
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('target column encoded')

        # remove duplicates 
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # apply text transformation to text column 

        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('text column transformed')
        return df
    except KeyError as e:
        logger.error("column not found: %s", e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s',e)
        raise

def main(text_column = 'text',target_column = 'target'):
    # load data, process it and save it 
    try:
        # fetch data from raw/data 
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')

        # transform the data 

        train_processed_data = preprocess_df(train_data, text_column,target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # save data in data/processed 

        data_path = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)


        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug("processed data saved to: %s",data_path)
    except FileNotFoundError as e:
        logger.error("File not Found: %s",e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s",e)
    except Exception as e:
        logger.error('failed to complete data transform process: %s',e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

