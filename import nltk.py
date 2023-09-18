import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


# Step 1: Read the Moby Dick file
filename = "/Users/wsy_956559_/nltk_data/corpora/gutenberg/melville-moby_dick.txt"
with open(filename, "r") as file:
    text = file.read()

# Step 2: Tokenization
tokens = word_tokenize(text)

# Step 3: Stop-words filtering
stop_words = set(stopwords.words("english"))
filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

# Step 4: Parts-of-Speech (POS) tagging
tagged_tokens = pos_tag(filtered_tokens)

# Step 5: POS frequency
pos_freq = FreqDist(tag for word, tag in tagged_tokens)
common_pos = pos_freq.most_common(5)
print("Most common parts of speech:")
for pos, count in common_pos:
    print(pos, count)

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()
wn_tags = {
    'N': 'n',
    'J': 'a',
    'V': 'v',
    'R': 'r'
}

lemmatized_tokens = []
for word, tag in tagged_tokens[:20]:
    wn_tag = wn_tags.get(tag[0])
    if wn_tag is not None:
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wn_tag))
    else:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))

print("\nLemmatized Tokens:")
print(lemmatized_tokens)

# Step 7: Plotting frequency distribution
pos_freq.plot()

# Display the plot
plt.show()