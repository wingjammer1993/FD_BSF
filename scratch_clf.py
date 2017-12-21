import sklearn
from sklearn import datasets
from os import listdir
from os.path import isfile, join
import io
from sklearn import svm
import collections
import NB
import scipy as sp
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk


def predict_classify(category_path):
    ip_data = sklearn.datasets.load_files(category_path)
    acc_baseline = []
    acc_sentiment = []

    for i in range(50):
        x_train, x_test, y_train, y_test = train_test_split(ip_data.data, ip_data.target, train_size=0.85)
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3))
        training_vector = vectorizer.fit_transform(x_train)
        test_vector = vectorizer.transform(x_test)

        # BASELINE
        #classifier = svm.LinearSVC()
        classifier = sklearn.linear_model.LogisticRegression()
        classifier.fit(training_vector, y_train)
        prediction = classifier.predict(test_vector)
        accuracy = sklearn.metrics.accuracy_score(y_test, prediction)
        acc_baseline.append(accuracy)

        # SENTIMENT
        sentiment = []
        sid = SentimentIntensityAnalyzer()
        for sentence in ip_data.data:
            ss = sid.polarity_scores(sentence)
            if ss['neg'] > ss['pos']:
                sentiment.append(0)
            else:
                sentiment.append(1)

        sentiment_train = sentiment[0:len(x_train)]
        sentiment_test = sentiment[len(x_train)-1:-1]

        training_vector = sp.sparse.hstack((training_vector, csr_matrix(sentiment_train).T))
        test_vector = sp.sparse.hstack((test_vector, csr_matrix(sentiment_test).T))

        #classifier = svm.LinearSVC()
        classifier = sklearn.linear_model.LogisticRegression()
        classifier.fit(training_vector, y_train)
        prediction = classifier.predict(test_vector)
        accuracy = sklearn.metrics.accuracy_score(y_test, prediction)
        acc_sentiment.append(accuracy)

    print(acc_baseline)
    print(acc_sentiment)
    print(np.array(acc_baseline).mean())
    print(np.array(acc_sentiment).mean())


if __name__ == "__main__":

    path_ip = 'C:\\Users\\Amruta\\PycharmProjects\\FD_BSF\\kfold'
    path_test = 'C:\\Users\\Amruta\\PycharmProjects\\FD_BSF\\test'
    gold_pos = r'hotelT-train.txt'
    gold_neg = r'hotelF-train.txt'
    dev_set = r'devset.txt'
    output_file = 'output_SVM.txt'
    predict_classify(path_ip)







