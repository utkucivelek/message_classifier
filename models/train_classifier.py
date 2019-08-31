import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt', 'wordnet', 'stopwords'])
stop_words = stopwords.words("english")


def load_data(database_filepath):
    """Loads cleaned data from SQL, sets corresponding values
    as input variable and target variables"""
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table("DisasterMessages", engine)
    
    X = df["message"]   #Raw messages
    Y = df.iloc[:,4:39] #Category values
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """Tokenizes and lemmatizes provide text"""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    """Generates a ML pipeline which uses Grid Search for tuning
    Random Forest Classifier."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [5, 10]}
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Prints out precision, recall, and Fscore of a model,
    by categories and total average""" 
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)-1):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
    
    #Printing the average accuracy values of all categories
    Y_result = []
    for i in range(Y_pred.shape[1]):
        Y_result.append(precision_recall_fscore_support(Y_test.iloc[:,i], Y_pred[:,i], average='weighted'))
    Y_result = pd.DataFrame(Y_result)
    Y_result.columns=["Total_Precision","Total_Recall","Total_Fscore","None"]
    Y_result.drop(["None"], axis=1, inplace=True)
    print(Y_result.mean(axis=0))

    
def save_model(model, model_filepath):
    """Saves the model as pickle file into specified path"""
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
