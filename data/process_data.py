import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine

''' Loads and merges the messages and categories data from csv files.
Args:
    messages_filepath: string, csv file path for the messages
    categories_filepath: string, csv file path for the categories
Returns:
    df: DataFrame, merged dataframe of the messages and categories datasets
'''
def load_data(messages_filepath,categories_filepath):
    
    '''We will read two dataframe and merge them on id column
    
       Our output will be the new dataframe called df which merged of messages and categories.
    '''
    
    # READ DATASETS
    messages_df=pd.read_csv(messages_filepath)
    categories_df=pd.read_csv(categories_filepath)
    
    # MERGE DATASETS
    df= messages_df.merge(categories_df, on='id')
    
    # RETURN OUTPUT
    
    return df

'''Preprocesses dataframe to extract categories into separate columns.
Args:
    df: DataFrame, dataframe to be preprocessed
Returns:
    df: DataFrame, preprocessed dataframe
'''
def clean_data(df):
    
    categories=df['categories'].str.split(';',expand=True)
    
    category_colnames=categories.loc[0].str.split('-',expand=True)[0].tolist()
    
    categories.columns=category_colnames
    
    for i in categories:        
        categories[i]  = pd.to_numeric(categories[i].str[-1])
        
    categories['id']=df['id']
    
    df=df.merge(categories,on='id')
    
    
    df.drop('categories',axis=1,inplace=True)
    
    df = df.drop_duplicates( keep='first')
    
    return df
    

def save_data(df, database_filename):
    
    """
    Saves given dataframe into an table in SQLite database file.
    Input:
    - df: DataFrame <- Pandas DataFrame containing cleaned data of messages and categories
    - database_filename: String <- Location of file where the database file is to be stored    
    """
    # Create connection with database
    engine = create_engine('sqlite:///'+ database_filename)
    
    # Save dataset to database table
    df.to_sql('DisasterResponse', engine, if_exists = 'replace', index=False)

def main():
    
    if len(sys.argv) == 4:
        
        messages_filepath, categories_filepath, database_filename = sys.argv[1:]
    
    
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
    
        print('Cleaning data...')
        df = clean_data(df)
    
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)
    
        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()