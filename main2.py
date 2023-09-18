import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#  Read the Moby Dick file
filename = "/Users/wsy_956559_/nltk_data/corpora/gutenberg/melville-moby_dick.txt"
with open(filename, "r") as file:
    text = file.read()

# Tokenization
tokenizer = nltk.tokenize.sent_tokenize
sentences = tokenizer(text)

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
scores = [analyzer.polarity_scores(sentence)["compound"] for sentence in sentences]
average_score = sum(scores) / len(scores)

# Determine overall text sentiment
if average_score > 0.05:
    overall_sentiment = "positive"
else:
    overall_sentiment = "negative"

# Display results
print("Average Sentiment Score:", average_score)
print("Overall Text Sentiment:", overall_sentiment)