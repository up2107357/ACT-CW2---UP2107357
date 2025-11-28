print("cd first")
#pip install kaggle
#kaggle datasets download hadifariborzi/amazon-books-dataset-20k-books-727k-reviews
#tar -xf amazon_books_metadata_sample_20k.zip
import zipfile
import pandas as pd
import os

zip_path = "amazon-books-dataset-20k-books-727k-reviews.zip"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("dataset")

print(oslistdir("dataset"))
csv_path = "dataset/file.csv"
df = pd.read_csv(csv_path)
print(df.head())
