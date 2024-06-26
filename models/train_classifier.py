import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')


''' Loads data from a given database.
Args:
    database_filepath: string, the database path for the data to be loaded
Returns:
    X: DataFrame, features dataset
    Y: DataFrame, target dataset
    categories: Index, target class names
'''
# LOAD DATA FUNCTION


def load_data(database_filepath):
    
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    
    X=df['message']
    y=df.iloc[:,4:]
    category_names = y.columns.tolist()
    
    return X,y,category_names
    
    

# TOKENIZING DATA FUNCTION


def tokenize_data(text):
    
    new_text=re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words=word_tokenize(new_text)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    tokens= [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return tokens

    

def build_model():
    
    pipeline=Pipeline([

    ('vect',CountVectorizer(tokenizer=tokenize_data)),
    ('tfidf',TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(random_state=101)))
        
    ])
    
    parameter_grid = {
#        'vect__max_df': (0.5, 0.75, 1.0),
#        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [ 10,15,20],
       'clf__estimator__min_samples_split': [2, 3],
    }

    model=GridSearchCV(estimator=pipeline, param_grid=parameter_grid,n_jobs=1,verbose=3)


    return model



def estimate_model(model, X_test, y_test):
    
    y_predictions=model.predict(X_test)
    for i,col in enumerate(y_test.columns):
        
        print(col,'--------------------')
        print(classification_report(y_test.iloc[:, i], y_predictions[:, i]))
        
        
def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))
    
    #'wb' means 'write binary' 
    # writes the pickeled data into a file. 
    
    
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
    
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        estimate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        


if __name__ == '__main__':
    main()
    