import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# function to extract nouns from a given text
def extract_nouns(text):
    nouns = []
    words = word_tokenize(text)
    
    tagged_words = pos_tag(words)

    # extract the nouns (NN, NNS, NNP, NNPS) from tagged words
    for word, tag in tagged_words:
        if re.match(r'^NN', tag):
            nouns.append(word)
    return nouns


# Dictionary to store noun frequencies
noun_freq = {}
scene = "beach"

# read the content of the description.txt file and strip all the brackets (counts brackets as nouns sometimes if not)
with open(f'memory/{scene}_description.txt', 'r') as file:
    descriptions = [re.sub(r'[\[\]]', '', line) for line in file]


listnouns = []

# extract the nouns and their frequencies from each description
for description in descriptions:
    listnouns.append(extract_nouns(description))


# write sorted noun frequencies to a text file in a 2D list format
with open(f'memory/{scene}_features.txt', 'w') as outfile:

    for nouns in listnouns:
        for noun in nouns:
            outfile.write(noun + " ")


print('completed')
