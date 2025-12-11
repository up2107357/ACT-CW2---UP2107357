def get_data():
    import pandas as pd
    import re # To clean up the strings before the publishing date
    #The metadata is much more valuable for me to analyse, but there are columns that are not useful in any way.
    df_1 = pd.read_csv("kaggle/amazon_books_metadata_sample_20k.csv")

    # Removing the columns that provide nothing to my project. Credit to ChatGPT which recommended the below line of code:
    useless_cols = ['subtitle', 'author_about', 'features_text', 'dimensions', 'item_weight', 'bought_together', 'isbn_10', 'isbn_13', 'images', 'videos', 'store', 'parent_asin', 'description', 'price', 'main_category', 'category_level_1_main']
    df_metadata = df_1.drop(useless_cols, axis = 1)
    ''' 
    Due to publisher_date being set as Month Day, Year or 'Edition' (Month Day, Year . 
    Note, the above unclosed bracket is intentional as it is how it was presented wihtin the dataset.
    I will change it to just the year by collecting the four-digit year value. 
    Credit to ChatGPT which showed me how to do this seen via the code below. 
    '''
    # Step 1 — extract date inside brackets if present
    df_metadata['maybe_date'] = df_metadata['publisher_date'].str.extract(r'\((.*?)\)')

    # Step 2 — if missing, use the original text
    df_metadata['maybe_date'] = df_metadata['maybe_date'].fillna(df_metadata['publisher_date'])

    # Step 3 — extract ANY 4-digit year from the string
    df_metadata['publisher_date'] = df_metadata['maybe_date'].str.extract(r'((?:19|20)\d{2})', expand=False)

    # Step 4 — convert to numeric
    df_metadata['publisher_date'] = pd.to_numeric(df_metadata['publisher_date'], errors='coerce')

    return df_metadata

def test_train_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    

    df_metadata = get_data() # This retrives the cleaned data from above.
    le = LabelEncoder()
    df_metadata['author_id'] = le.fit_transform(df_metadata['author_name'])

    # Other contexts around each author:
    df_metadata['author_book_count'] = df_metadata.groupby('author_name')['title'].transform('count')
    df_metadata['debut'] = (df_metadata['author_book_count'] == 1).astype(int)

    # Accounting for the other numeric features within the dataset:
    numeric_metrics = ['author_book_count', 'debut', 'rating number', 'average_rating']
    x_num = df_metadata[numeric_metrics].fillna(0).values
    scaler = StandardScaler()
    x_num = scaler.fit_transform(x_num)

    # Now to deal with whether a title is a success or not, be it critical or commercial:
    earnings = np.log1p(df_metadata['rating_number'])
    reviews = df_metadata['average_rating']
    impact = 0.6 * earnings + 0.4 * reviews
    df_metadata['is_supported'] = (impact > impact.quantile(0.7)).astype(int)

    x_author = df_metadata['author_id'].values
    y = df_metadata['is_supported'].values

    # Performing the test-train split:
    X_author_train, X_author_test, X_num_train, X_num_test, y_train, y_test = train_test_split(x_author, x_num, y, test_size = 0.2, random_state = 42)
   
    return X_author_train, X_author_test, X_num_train, X_num_test, y_train, y_test