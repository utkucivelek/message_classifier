import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """loads message and catgories data in two files, returns merge of them""" 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    return df


def clean_data(df):
    """Splits the categories data, existing in a single column, to dedicated columns
    Sets column names accordingly, removes duplicates, fills NaNs with zeros""" 
    df_categories = pd.DataFrame(df["categories"].str.split(";", expand=True))
    row = df_categories.iloc[0]
    category_colnames = row.str.split('-').str[0].tolist()
    df_categories.columns = category_colnames
    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].str[-1:]
        # convert column from string to numeric
        df_categories[column] = pd.to_numeric(df_categories[column])
    df = df.drop(columns=["categories"])
    df = pd.concat([df,df_categories], axis=1)
    df_clean = df.drop_duplicates()
    df_clean = df_clean.fillna("0")
    return df_clean


def save_data(df, database_filename):
    """Saves cleaned data to SQL, with the provided name"""
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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
