import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\d', ' ', text)  
    text = text.strip()
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    try:
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    except Exception as e:
        print(f"Error in lemmatization: {e}")
        return text


def load_data(file_path, has_labels=True):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if has_labels:
                if ':::' in line:
                    parts = line.strip().split(' ::: ')
                    if len(parts) == 4:
                        plot = parts[3] 
                        genre = parts[2]  
                        data.append(plot)
                        labels.append(genre)
            else:
                if ':::' in line:
                    parts = line.strip().split(' ::: ')
                    if len(parts) == 3:
                        plot = parts[2]  
                        data.append(plot)

    return data, labels if has_labels else data


train_plots, train_genres = load_data('train_data.txt')
test_plots, _ = load_data('test_data.txt', has_labels=False)
_, test_genres = load_data('test_data_solution.txt')


train_data = pd.DataFrame({'plot': train_plots, 'genre': train_genres})
test_data = pd.DataFrame({'plot': test_plots, 'genre': test_genres})

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)


train_data['cleaned_plot'] = train_data['plot'].apply(preprocess_text).apply(lemmatize_text)
test_data['cleaned_plot'] = test_data['plot'].apply(preprocess_text).apply(lemmatize_text)


tfidf_vectorizer = TfidfVectorizer(max_features=30000, stop_words=stopwords.words('english'), ngram_range=(1, 2))
X_train = tfidf_vectorizer.fit_transform(train_data['cleaned_plot'])
X_test = tfidf_vectorizer.transform(test_data['cleaned_plot'])


y_train = train_data['genre']
y_test = test_data['genre']


nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_train, y_train)
y_test_pred = nb_classifier.predict(X_test)


print("\nNaive Bayes Test Results:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))


predictions_df = pd.DataFrame({'plot': test_plots, 'predicted_genre': y_test_pred})
predictions_df.to_csv('predictions.csv', index=False)

print("\nPredictions saved to predictions.csv")
