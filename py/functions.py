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

    df_metadata = get_data() # This retrives the cleaned data from above.
    X = df_metadata['average_rating']
    Y = df_metadata['price_numeric'].values*df_metadata['rating_number'].values 