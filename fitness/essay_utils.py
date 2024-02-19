import numpy as np
import nltk
import re
from nltk.corpus import stopwords

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def make_feature_vec(words, embeddings, num_features):
    """Cria um vetor de recursos a partir da lista de palavras de um ensaios usando um dicionário de embeddings."""
    feature_vec = np.zeros((num_features,), dtype="float32")
    num_words =  0.
    for word in words:
        if word in embeddings:
            num_words +=  1
            feature_vec = np.add(feature_vec, embeddings[word])       
    feature_vec = np.divide(feature_vec, num_words)
    return feature_vec


def get_avg_feature_vecs(essays, embeddings, num_features):
    """Main function to generate the word vectors for any embedding model."""
    counter = 0
    essay_feature_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essay_feature_vecs[counter] = make_feature_vec(essay, embeddings, num_features)
        counter = counter + 1
    return essay_feature_vecs