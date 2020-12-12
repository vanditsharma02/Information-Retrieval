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
from tabulate import tabulate
import sys

path_data_directory = sys.argv[1]
# path_data_directory = './dataset'
output_file = sys.argv[2]
# output_file = 'Output File 2'

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

# print('Creating Document Vectors')
normalized_document_vectors = {}
normalized_document_array = np.empty((0, len(InvertedPositionalIndex1)), float)

for c in classes:
    path = path_data_directory + '/' + c + '/train'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        document_vector = np.zeros((1, len(InvertedPositionalIndex1)), float)
        counter = 0
        for key in InvertedPositionalIndex1:
            list = InvertedPositionalIndex1[key]
            flag = 0
            for f in range(len(list)):
                if list[f][0] == file:
                    flag = 1
                    break
            if flag == 1:
                document_vector[0, counter] = key[1]*list[f][1]
            else:
                document_vector[0, counter] = 0
            counter = counter + 1
        #print('here2')
        normalizing_factor = 0
        temp = np.square(document_vector)
        normalizing_factor = np.sum(temp)
        normalizing_factor = math.sqrt(normalizing_factor)
        if normalizing_factor != 0:
            normalized_document_vector = np.divide(document_vector, normalizing_factor)
        else:
            normalized_document_vector = document_vector

        normalized_document_vectors[file] = normalized_document_vector[0]
        normalized_document_array = np.append(normalized_document_array, normalized_document_vector, axis=0)

def train_rocchio():
    uc = {}
    for c in classes:
        uc[c] = []
        path = path_data_directory + '/' + c + '/train'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if len(uc[c]) != 0:
                sum = []
                vd = normalized_document_vectors[file]
                zip_object = zip(uc[c], vd)
                for uc_i, vd_i in zip_object:
                    sum.append(uc_i + vd_i)
                uc[c] = sum
            else:
                uc[c] = normalized_document_vectors[file]
        l = len(files)
        for i in range(len(uc[c])):
            uc[c][i] = float(uc[c][i])/l

    return uc

def apply_rocchio(vd, rocchio_centroids):
    distances = {}
    for c in classes:
        difference_c = []
        zip_object = zip(rocchio_centroids[c], vd)
        for uc_i, vd_i in zip_object:
            difference_c.append(uc_i - vd_i)
        distances[c] = magnitude(difference_c)

    # print(distances)
    temp = min(distances.values())
    res = [key for key in distances if distances[key] == temp]
    prediction = res[0]
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

    f1_score_1 = tp / (tp + 0.5 * (fp + fn))

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

    macro_avg_f1_score = (f1_score_1 + f1_score_2) / 2

    return macro_avg_f1_score

def magnitude(vec):
    summ = 0
    for i in range(len(vec)):
        summ = vec[i]*vec[i] + summ
    return pow(summ, 0.5)

def get_results(path_data_directory, output_file):
    b = 0
    actual = []
    prediction = []
    results = {}

    rocchio_centroids = train_rocchio()

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
            lemmatized_tokens = set(filtered_tokens)

            document_vector = np.zeros((1, len(InvertedPositionalIndex1)), float)
            counter = 0

            for key in InvertedPositionalIndex1:
                if key[0] in lemmatized_tokens:
                    document_vector[0, counter] = key[1] * filtered_tokens.count(key[0])
                else:
                    document_vector[0, counter] = 0
                counter = counter + 1

            normalizing_factor = 0
            temp = np.square(document_vector)
            normalizing_factor = np.sum(temp)
            normalizing_factor = math.sqrt(normalizing_factor)
            if normalizing_factor != 0:
                normalized_document_vector = np.divide(document_vector, normalizing_factor)
            else:
                normalized_document_vector = document_vector

            actual.append(c)
            prediction.append(apply_rocchio(normalized_document_vector[0], rocchio_centroids))

    results[b] = (calculate_f1(actual, prediction))
    x = results.keys()
    y = []
    for key in x:
        y.append(results[key])

    # plt.plot(x, y, marker='o', color='b')
    # plt.title('Effect of b on F1 score (Rocchio)')
    # plt.xlabel('b')
    # plt.ylabel('F1 score')
    # plt.show()

    table = [['Rocchio', y[0]]]
    headers = ['b', 0]

    result = output_file
    result = result + '\n' + tabulate(table, headers, tablefmt="plain")

    path = './' + output_file + '.txt'
    f = open(path, mode='w', encoding='utf-8')
    f.write(result)
    f.close()

get_results(path_data_directory, output_file)