# Disaster Response Pipeline Project
This is a project prepared in the context of Udacity Data Scientist Nano-program.

Most of the codes & explainations are provided from the program.

In this repository there exist three code files:

### process_data.py as ETL Pipeline
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database with sqlalchemy library

### train_classifier.py as ML Pipeline
- Loads data from the SQLite database with sqlalchemy library
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline with nltk & sklearn libraries
- Trains and tunes RandomForestClassifier using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### run.py as Web App
- Generates a web app with Flask & Plotly libraries


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Run another terminal windows and type env|grep WORK
In a web browser window, open https://SPACEID-3001.SPACEDOMAIN by changing UPPERCASE elements with the related outputs of env|grep WORK

### Example App:
https://view6914b2f4-3001.udacity-student-workspaces.com/
