# Import required libraries
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Filters stopwords and punctuation from the description
# for a more accurate cosine similarity


def d_filter(description):
    # Get list of stop words
    stop_words = set(stopwords.words('english'))

    # Tokenize the words
    word_tokens = word_tokenize(description)

    filtered_list = [word for word in word_tokens if not word.lower() in stop_words]

    filtered_description = ' '.join(filtered_list)

    filtered_description = filtered_description.translate(str.maketrans('', '', string.punctuation))

    return filtered_description

# Obtain generated description


gen_description_file = open('gen.txt', 'r')
gen_description = gen_description_file.read()
gen_description = d_filter(gen_description)

# Obtain feature list
feature_list_file = open('memory/beach_features.txt', 'r')
feature_list = feature_list_file.read()

# Vectorize the data
count_vect = CountVectorizer()
corpus = [gen_description, feature_list]
X_train_counts = count_vect.fit_transform(corpus)
pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names_out(), index=['gen_description', 'feature_list'])

vectorizer = TfidfVectorizer()
trsfm = vectorizer.fit_transform(corpus)
pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names_out(), index=['gen_description', 'feature_list'])

# Compute cosine similarity
cos_sim = cosine_similarity(trsfm[0:1], trsfm)[0][1]

print(cos_sim)
# Assess confidence
if cos_sim >= 0.5:
    print("Generator acceptable description")
else:
    print("Generator description failed")
