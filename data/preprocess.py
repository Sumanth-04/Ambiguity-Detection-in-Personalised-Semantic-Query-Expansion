import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)

# Load English articles
df = pd.read_csv("data/iwa_articles.csv")

print("Columns available:", df.columns.tolist())
print("Sample before cleaning:")
print(df["OCR"].head(2))  # OCR column has the article text

# Apply preprocessing
df["clean_text"] = df["OCR"].apply(preprocess_text)

print("\nSample after cleaning:")
print(df["clean_text"].head(2))

# Save
df.to_csv("data/iwa_preprocessed.csv", index=False)
print("\nSaved to data/iwa_preprocessed.csv")