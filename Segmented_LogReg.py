import sklearn
import io
from sklearn.model_selection import train_test_split
import NB


def get_gold_std(train_pos_sen, train_neg_sen):
    gold_set = {}
    test = io.open(train_pos_sen, 'r', encoding="utf8")
    for line in test.readlines():
        gold_set[str(line.split(None, 1)[0])] = (str(line.split(None, 1)[1]).strip(), 'T')
    test = io.open(train_neg_sen, 'r', encoding="utf8")
    for line in test.readlines():
        gold_set[str(line.split(None, 1)[0])] = (str(line.split(None, 1)[1]).strip(), 'F')
    return gold_set


def create_training_set(train_true, train_false, gold_set):

    train_positive = []
    train_negative = []
    pos_target = []
    neg_target = []
    test = io.open(train_true, 'r', encoding="utf8")
    for line in test.readlines():
        if gold_set[str(line.split(None, 1)[0])][0] == 'POS':
            train_positive.append(line)
            pos_target.append('T')
        else:
            train_negative.append(line)
            neg_target.append('T')

    test = io.open(train_false, 'r', encoding="utf8")
    for line in test.readlines():
        if gold_set[str(line.split(None, 1)[0])][0] == 'POS':
            train_positive.append(line)
            pos_target.append('F')
        else:
            train_negative.append(line)
            neg_target.append('F')

    return train_positive, pos_target, train_negative, neg_target


def predict_classify(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, train_size=0.85)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3))
    training_vector = vectorizer.fit_transform(x_train)
    test_vector = vectorizer.transform(x_test)
    classifier = sklearn.linear_model.LogisticRegression()
    classifier.fit(training_vector, y_train)
    prediction = classifier.predict(test_vector)
    accuracy = sklearn.metrics.accuracy_score(y_test, prediction)
    return accuracy


def predict_test(training, target, test):

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3))
    training_vector = vectorizer.fit_transform(training)
    test_vector = vectorizer.transform(test)
    classifier = sklearn.linear_model.LogisticRegression()
    classifier.fit(training_vector, target)
    prediction = classifier.predict(test_vector)

    verdict = {}
    for index, line in enumerate(test):
        if prediction[index] == 'T':
            verdict[str(line.split(None, 1)[0])] = 'T'
        else:
            verdict[str(line.split(None, 1)[0])] = 'F'

    return verdict


def create_test_set(teset, test_nb_sen):
    test_positive = []
    test_negative = []
    test = io.open(teset, 'r', encoding="utf8")
    for line in test.readlines():
        if test_nb_sen[str(line.split(None, 1)[0])] == 'POS':
            test_positive.append(line)
        else:
            test_negative.append(line)
    return test_positive, test_negative


def print_output(out_file, tesset, ver_1, ver_2):
    op = open(out_file, 'w')
    ip = open(tesset, 'r')
    for line in ip.readlines():
        op_key = str(line.split(None, 1)[0])
        if op_key in ver_1:
            printable = str(op_key) + '\t' + str(ver_1[op_key]) + '\n'
            op.write(printable)
        elif op_key in ver_2:
            printable = str(op_key) + '\t' + str(ver_2[op_key]) + '\n'
            op.write(printable)


if __name__ == "__main__":

    train_pos = r'output_NB_true.txt'
    train_neg = r'output_NB_false.txt'
    train_tru = r'hotelT-train.txt'
    train_fals = r'hotelF-train.txt'
    test_set = r'finaltest.txt'
    train_NB_pos = r'hotelPosT-train.txt'
    train_NB_neg = r'hotelNegT-train.txt'
    outfile = r'Output_crazy.txt'
    gs = get_gold_std(train_pos, train_neg)
    tp, p_tar, tn, n_tar = create_training_set(train_tru, train_fals, gs)
    test_NB = NB.test_naive_bayes(train_NB_pos, train_NB_neg, test_set, 1)
    test_pos, test_neg = create_test_set(test_set, test_NB)
    verdict_1 = predict_test(tp, p_tar, test_pos)
    verdict_2 = predict_test(tp, p_tar, test_neg)
    print_output(outfile, test_set, verdict_1, verdict_2)
    # count = 0
    # ip = open(test_set, 'r')
    # for line in ip.readlines():
    #     if str(line.split(None, 1)[0]) in verdict_1:
    #         if verdict_1[str(line.split(None, 1)[0])] == gs[str(line.split(None, 1)[0])][-1]:
    #             count += 1
    #     if str(line.split(None, 1)[0]) in verdict_2:
    #         if verdict_2[str(line.split(None, 1)[0])] == gs[str(line.split(None, 1)[0])][-1]:
    #             count += 1
    #
    # length = (len(verdict_1)+len(verdict_2))
    # print(count/float(length))















