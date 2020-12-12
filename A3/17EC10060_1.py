from os import listdir
from os.path import isfile, join
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import sys
from tabulate import tabulate

path_data_directory = sys.argv[1]
# path_data_directory = './dataset'
output_file = sys.argv[2]
# output_file = 'Output File 1'

classes = ['class1', 'class2']
datasets = ['train', 'test']

N_total = 0
InvertedPositionalIndex = {}
for c in classes:
    path = path_data_directory + '/' + c + '/train'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    N_total = N_total + len(files)
    for file in files:
        f = open(path + '/' + file, mode='r')
        content = f.read()
        f.close()
        translator = str.maketrans('', '', string.punctuation)
        content = content.translate(translator)
        tokens = word_tokenize(content)
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = lemmatized_tokens
        stop_words = set(stopwords.words('english'))
        punctuation_removed = [stop_word.translate(translator) for stop_word in stop_words]
        lemmatized_stopwords = [lemmatizer.lemmatize(stop_word) for stop_word in punctuation_removed]
        filtered_tokens = [token for token in tokens if not token in lemmatized_stopwords]
        lemmatized_tokens = set(filtered_tokens)
        for lemmatized_token in lemmatized_tokens:
            try:
                InvertedPositionalIndex[lemmatized_token][c]['df'] = InvertedPositionalIndex[lemmatized_token][c]['df'] + 1
                InvertedPositionalIndex[lemmatized_token][c]['tf'].append((file, filtered_tokens.count(lemmatized_token)))
            except KeyError:
                try:
                    InvertedPositionalIndex[lemmatized_token]
                    InvertedPositionalIndex[lemmatized_token][c] = {}
                    InvertedPositionalIndex[lemmatized_token][c]['df'] = 1
                    InvertedPositionalIndex[lemmatized_token][c]['tf'] = []
                    InvertedPositionalIndex[lemmatized_token][c]['tf'].append((file, filtered_tokens.count(lemmatized_token)))
                except:
                    InvertedPositionalIndex[lemmatized_token] = {}
                    InvertedPositionalIndex[lemmatized_token][c] = {}
                    InvertedPositionalIndex[lemmatized_token][c]['df'] = 1
                    InvertedPositionalIndex[lemmatized_token][c]['tf'] = []
                    InvertedPositionalIndex[lemmatized_token][c]['tf'].append((file, filtered_tokens.count(lemmatized_token)))

InvertedPositionalIndex1 = {}
for key in InvertedPositionalIndex:
    df = 0
    for c in classes:
        try:
            df = df + InvertedPositionalIndex[key][c]['df']
        except:
            df = df + 0
    idf = math.log10(N_total/df)
    InvertedPositionalIndex1[(key, idf)] = []
    for c in classes:
        try:
            for entry in InvertedPositionalIndex[key][c]['tf']:
                tf = math.log10(1 + entry[1])
                InvertedPositionalIndex1[(key, idf)].append((entry[0], tf))
        except:
            pass

def get_features(c, x):
    mutual_information = {}
    path = path_data_directory + '/' + c + '/train'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    class_docs = len(files)
    nonclass_docs = N_total - class_docs

    for key in InvertedPositionalIndex:
        # try:
        #     present = InvertedPositionalIndex[key][c]
        N10 = float(0)
        for cl in classes:
            if cl != c:
                try:
                    N10 = N10 + InvertedPositionalIndex[key][cl]['df']
                except:
                    N10 = N10 + 0
        try:
            N11 = float(InvertedPositionalIndex[key][c]['df'])
        except:
            N11 = float(0)

        N00 = nonclass_docs - N10
        N01 = class_docs - N11

        N1_ = N10 + N11
        # print('N1_ : ' + str(N1_))
        N_1 = N01 + N11
        # print('N_1 : ' + str(N_1))
        N0_ = N00 + N01
        # print('N0_ : ' + str(N0_))
        N_0 = N00 + N10
        # print('N_0 : ' + str(N_0))
        # print()
        N = N00 + N01 + N10 + N11
        I = 0
        if N != 0 and N11 != 0:
            I = I + (N11/N)*math.log2(N*N11/(N1_*N_1))
        else:
            I = I + 0
        if N != 0 and N01 != 0:
            I = I + (N01/N)*math.log2(N*N01/(N0_*N_1))
        else:
            I = I + 0
        if N != 0 and N10 != 0:
            I = I + (N10/N)*math.log2(N*N10/(N1_*N_0))
        else:
            I = I + 0
        if N != 0 and N00 != 0:
            I = I + (N00/N)*math.log2(N*N00/(N0_*N_0))
        else:
            I = I + 0
        mutual_information[key] = I
        # except:
        #     continue

    k = Counter(mutual_information)
    features = k.most_common(x)
    temp = {}
    for entry in features:
        temp[entry[0]] = entry[1]
    features = temp

    return features

def train_multinomial_naive_bayes(features):
    multinomial_naive_bayes = {}
    for c in classes:
        multinomial_naive_bayes[c] = {}
        multinomial_naive_bayes[c]['c'] = {}
        multinomial_naive_bayes[c]['cBAR'] = {}
        total_cterms = 0
        for term in InvertedPositionalIndex:
            if c in InvertedPositionalIndex[term]:
                for entry in InvertedPositionalIndex[term][c]['tf']:
                    total_cterms = total_cterms + entry[1]

        total_cBARterms = 0
        for c1 in classes:
            if c1 != c:
                for term in InvertedPositionalIndex:
                    if c1 in InvertedPositionalIndex[term]:
                        for entry in InvertedPositionalIndex[term][c1]['tf']:
                            total_cBARterms = total_cBARterms + entry[1]

        num_terms = len(InvertedPositionalIndex)

        for feature in features[c]:
            total_freq_c = 0
            try:
                for entry in InvertedPositionalIndex[feature][c]['tf']:
                    total_freq_c = total_freq_c + entry[1]
            except:
                total_freq_c = 0

            total_freq_cBAR = 0
            for c1 in classes:
                if c1 != c:
                    try:
                        for entry in InvertedPositionalIndex[feature][c1]['tf']:
                            total_freq_cBAR = total_freq_cBAR + entry[1]
                    except:
                        total_freq_cBAR = 0

            multinomial_naive_bayes[c]['c'][feature] = float(total_freq_c + 1) / float(total_cterms + num_terms)
            multinomial_naive_bayes[c]['cBAR'][feature] = float(total_freq_cBAR + 1) / float(total_cBARterms + num_terms)

    return multinomial_naive_bayes

def apply_multinomial_naive_bayes(counts, tokens, multinomial_naive_bayes, features, Pc, PcBAR):
    Pcdoc = {}
    PcBARdoc = {}

    for c in classes:
        Pcdoc[c] = math.log10(Pc[c])
        PcBARdoc[c] = math.log10(PcBAR[c])

        for token in tokens:
            if token in features[c]:
                Pcdoc[c] = Pcdoc[c] + counts[token]*math.log10(multinomial_naive_bayes[c]['c'][token])
                PcBARdoc[c] = PcBARdoc[c] + counts[token]*math.log10(multinomial_naive_bayes[c]['cBAR'][token])

    k = Counter(Pcdoc)
    temp = k.most_common(1)
    prediction = temp[0][0]
    return prediction

def train_bernoulli_naive_bayes(features):
    bernoulli_naive_bayes = {}
    for c in classes:
        bernoulli_naive_bayes[c] = {}
        bernoulli_naive_bayes[c]['c'] = {}
        bernoulli_naive_bayes[c]['cBAR'] = {}

        path = path_data_directory + '/' + c + '/train'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        class_docs = len(files)
        nonclass_docs = N_total - class_docs

        for feature in features[c]:
            total_freq_c = 0
            try:
                total_freq_c = len(InvertedPositionalIndex[feature][c]['tf'])
            except:
                total_freq_c = 0

            total_freq_cBAR = 0
            for c1 in classes:
                if c1 != c:
                    try:
                        total_freq_cBAR = total_freq_cBAR + len(InvertedPositionalIndex[feature][c1]['tf'])
                    except:
                        total_freq_cBAR = 0

            bernoulli_naive_bayes[c]['c'][feature] = float(total_freq_c + 1) / float(class_docs + 2)
            bernoulli_naive_bayes[c]['cBAR'][feature] = float(total_freq_cBAR + 1) / float(nonclass_docs + 2)

    return bernoulli_naive_bayes

def apply_bernoulli_naive_bayes(tokens, bernoulli_naive_bayes, features, Pc, PcBAR):
    Pcdoc = {}
    PcBARdoc = {}

    for c in classes:
        Pcdoc[c] = math.log10(Pc[c])
        PcBARdoc[c] = math.log10(PcBAR[c])
        for term in InvertedPositionalIndex:
            if term in tokens and term in features[c]:
                Pcdoc[c] = Pcdoc[c] + math.log10(bernoulli_naive_bayes[c]['c'][term])
                PcBARdoc[c] = PcBARdoc[c] + math.log10(bernoulli_naive_bayes[c]['cBAR'][term])
            elif term not in tokens and term in features[c]:
                Pcdoc[c] = Pcdoc[c] + math.log10(1 - bernoulli_naive_bayes[c]['c'][term])
                PcBARdoc[c] = PcBARdoc[c] + math.log10(1 - bernoulli_naive_bayes[c]['cBAR'][term])

    k = Counter(Pcdoc)
    temp = k.most_common(1)
    prediction = temp[0][0]
    return prediction

def calculate_f1(actual, prediction):

    tp = float(0)
    tn = float(0)
    fp = float(0)
    fn = float(0)

    for i in range(len(actual)):
        if actual[i] == classes[1] and prediction[i] == classes[1]:
            tn = tn + 1
        elif actual[i] == classes[0] and prediction[i] == classes[1]:
            fn = fn + 1
        elif actual[i] == classes[1] and prediction[i] == classes[0]:
            fp = fp + 1
        elif actual[i] == classes[0] and prediction[i] == classes[0]:
            tp = tp + 1

    f1_score_1 = tp/(tp + 0.5*(fp + fn))

    tp = float(0)
    tn = float(0)
    fp = float(0)
    fn = float(0)

    for i in range(len(actual)):
        if actual[i] == classes[0] and prediction[i] == classes[0]:
            tn = tn + 1
        elif actual[i] == classes[1] and prediction[i] == classes[0]:
            fn = fn + 1
        elif actual[i] == classes[0] and prediction[i] == classes[1]:
            fp = fp + 1
        elif actual[i] == classes[1] and prediction[i] == classes[1]:
            tp = tp + 1

    f1_score_2 = tp / (tp + 0.5 * (fp + fn))

    macro_avg_f1_score = (f1_score_1 + f1_score_2)/2

    return macro_avg_f1_score

def get_results(path_data_directory, output_file):
    x_range = [1, 10, 100, 1000, 10000]
    results_multinomial = {}
    results_bernoulli = {}
    for x in x_range:
        actual = []
        Pc = {}
        PcBAR = {}
        features = {}
        for c in classes:
            features[c] = get_features(c, x)
            path = path_data_directory + '/' + c + '/train'
            files = [f for f in listdir(path) if isfile(join(path, f))]
            class_docs = len(files)
            nonclass_docs = N_total - class_docs
            Pc[c] = float(class_docs) / float(N_total)
            PcBAR[c] = float(nonclass_docs) / float(N_total)

        multinomial_naive_bayes = train_multinomial_naive_bayes(features)
        prediction_multinomial = []
        bernoulli_naive_bayes = train_bernoulli_naive_bayes(features)
        prediction_bernoulli = []

        for c in classes:
            path = path_data_directory + '/' + c + '/test'
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for file in files:
                f = open(path + '/' + file, mode='r')
                content = f.read()
                f.close()
                translator = str.maketrans('', '', string.punctuation)
                content = content.translate(translator)
                tokens = word_tokenize(content)
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                tokens = lemmatized_tokens
                stop_words = set(stopwords.words('english'))
                punctuation_removed = [stop_word.translate(translator) for stop_word in stop_words]
                lemmatized_stopwords = [lemmatizer.lemmatize(stop_word) for stop_word in punctuation_removed]
                filtered_tokens = [token for token in tokens if not token in lemmatized_stopwords]
                counts = {x:filtered_tokens.count(x) for x in filtered_tokens}
                lemmatized_tokens = set(filtered_tokens)
                actual.append(c)
                prediction_multinomial.append(apply_multinomial_naive_bayes(counts, lemmatized_tokens, multinomial_naive_bayes, features, Pc, PcBAR))
                prediction_bernoulli.append(apply_bernoulli_naive_bayes(lemmatized_tokens, bernoulli_naive_bayes, features, Pc, PcBAR))

        results_multinomial[x] = calculate_f1(actual, prediction_multinomial)
        results_bernoulli[x] = calculate_f1(actual, prediction_bernoulli)

    x = results_multinomial.keys()
    y = []
    for key in x:
        y.append(results_multinomial[key])

    # plt.plot(x, y, marker='o', color='b')
    # plt.title('Effect of feature selection on F1 score (Multinomial Naive Bayes)')
    # plt.xlabel('Number of features selected')
    # plt.ylabel('F1 score')
    # plt.show()

    z = []
    for key in x:
        z.append(results_bernoulli[key])

    # plt.plot(x, z, marker='o', color='b')
    # plt.title('Effect of feature selection on F1 score (Bernoulli Naive Bayes)')
    # plt.xlabel('Number of features selected')
    # plt.ylabel('F1 score')
    # plt.show()

    table = [['MultinomialNB', y[0], y[1], y[2], y[3], y[4]], ['BernoulliNB', z[0], z[1], z[2], z[3], z[4]]]
    headers = ['NumFeature', 1, 10, 100, 1000, 10000]

    result = output_file
    result = result + '\n' + tabulate(table, headers, tablefmt="plain")

    path = './' + output_file + '.txt'
    f = open(path, mode='w', encoding='utf-8')
    f.write(result)
    f.close()

get_results(path_data_directory, output_file)
