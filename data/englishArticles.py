import pandas as pd

df = pd.read_csv("data/iwa_articles.csv")

print("Total articles:", len(df))
print("Language values:", df["language"].unique())

# Filter English articles
df_en = df[df["language"] == "Anglais"]

print(f"English articles: {len(df_en)}")

df_en.to_csv("data/iwa_english.csv", index=False)
print("Saved to data/iwa_english.csv")