import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re

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

# read the content of the description.txt file and strip all the brackets (counts brackets as nouns sometimes if not)
with open('description.txt', 'r') as file:
    descriptions = [re.sub(r'[\[\]]', '', line) for line in file]

# extract the nouns and their frequencies from each description
for description in descriptions:
    nouns = extract_nouns(description)
    for noun in nouns:
        noun_freq[noun] = noun_freq.get(noun, 0) + 1

# sort noun frequencies by frequency in descending order
sorted_noun_freq = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)

# write sorted noun frequencies to a text file in a 2D list format
with open('features.txt', 'w') as outfile:
    outfile.write("[")

    for i, (noun, freq) in enumerate(sorted_noun_freq):
        if i < len(sorted_noun_freq) - 1:
            outfile.write(f"[{noun}, {freq}],")
        else:
            outfile.write(f"[{noun}, {freq}]")

    outfile.write("]")


print('completed')
