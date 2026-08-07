from datasets import load_dataset
import pandas as pd
import os

os.environ["HF_TOKEN"] = "your_token_here"  # paste your token here

os.makedirs("data", exist_ok=True)

dataset = load_dataset("fmadore/islam-west-africa-collection", "articles", token=os.environ["HF_TOKEN"])

df = dataset["train"].to_pandas()

print(df.shape)
print(df.columns.tolist())
print(df.head(3))

df.to_csv("data/iwa_articles.csv", index=False)
print("Saved to data/iwa_articles.csv")