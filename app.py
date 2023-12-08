import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List
from glovpy import GloVe
from embedding_explorer import show_network_explorer
from embedding_explorer import show_clustering

nltk.download("punkt")
nltk.download("stopwords")

english_stopwords = set(stopwords.words("english"))

# Load a proper database and column
try:
    database = pd.read_csv("koha.csv")  # replace with actual dataset
    data = database["date"].astype(str)  # replace with actual column name
    print("successfully loaded the database")
except Exception as e:
    print(f"problem with loading database: {e}")


# a function that tokenizes text in the column
def tokenize(text: str) -> List[str]:
    # Check if the text is a non-empty string
    if isinstance(text, str) and text.strip():
        tokens = word_tokenize(text)
        tokens = [
            token.lower()
            for token in tokens
            if token.isalpha() and token.lower() not in english_stopwords
        ]
        return tokens
    else:
        return []


try:
    # tokenize the dataset
    tokenized_data = [tokenize(text) for text in data]
    print(tokenized_data)
except Exception as e:
    print(f"error tokenizing: {e}")

# training word emnbeddings
model = GloVe(vector_size=25)
model.train(tokenized_data)

# query the word embeddign
print(model.wv.most_similar("serbs"))  # replace with actual keyword

# Static Word Embedding
vocabulary = model.wv.index_to_key
embeddings = model.wv.vectors
# show_network_explorer(vocabulary, embeddings=embeddings)

"""# Option 1: semantic relationship
# semantic relationship
show_network_explorer(vocabulary, embeddings)"""

# Dynamic Embedding Models 
show_clustering(embeddings=embeddings)
