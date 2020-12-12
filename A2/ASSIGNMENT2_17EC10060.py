from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
import bs4
import json
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pickle5 as pickle
import sys
from collections import Counter
import numpy as np

# Read HTML file and extract Text
files = [f for f in listdir('../Dataset/Dataset') if isfile(join('../Dataset/Dataset', f))]
ECTNestedDict = {}
TextDict = {}
#print('Converting HTML to Text')

for file in files:
    # Form ECTNestedDict similar to previous Assignment
    file_number = int(file.replace('.html',''))
    path = r'../Dataset/Dataset/%s' %(file)
    f = open(path, mode='r', encoding='utf-8')
    content = f.read()
    f.close()

    ECTNestedDict[file] = {}
    ECTNestedDict[file]['Date'] = ''

    soup = BeautifulSoup(content, features='html.parser')
    tags = soup.find_all('p')
    # tags = soup.find_all('p', attrs={'class' : 'p p1'})

    if len(tags) == 0:
        continue

    # print(file)
    text = []
    for x in tags[0]:
        if isinstance(x, bs4.element.NavigableString):
            text.append(x.strip())
    text = ' '.join(text)
    text = ' '.join(text.split('  '))
    text = text.split(' ')
    datentime = ' '.join(text[-6:-3])
    # print(datentime)
    ECTNestedDict[file]['Date'] = datentime

    participants = []
    strong_flag = 0
    for tag in tags:
        if tag.find('strong') != None:
            heading = " ".join(tag.find('strong').text.split())
            if heading == 'Company Participants' or heading == 'Conference Call Participants':
                strong_flag = 1
                continue
            else:
                break
        else:
            if strong_flag == 1:
                participants.append(tag.text)
    ECTNestedDict[file]['Participants'] = participants
    # print(participants)

    ECTNestedDict[file]['Presentation'] = {}
    speakers = []
    strong_flag = 0
    for tag in tags:
        if tag.find('strong') != None:
            heading = " ".join(tag.find('strong').text.split())
            if heading == 'Company Participants' or heading == 'Conference Call Participants':
                strong_flag = 0
                continue
            else:
                if heading != 'Question-and-Answer Session':
                    strong_flag = 1
                    speakers.append(heading)
                    try:
                        ECTNestedDict[file]['Presentation'][speakers[-1]]
                    except KeyError:
                        ECTNestedDict[file]['Presentation'][speakers[-1]] = []
                    continue
                else:
                    break
        else:
            if strong_flag == 1:
                # commment this if each para counts as spoken different times
                count = speakers.count(speakers[-1])
                try:
                    ECTNestedDict[file]['Presentation'][speakers[-1]][count - 1] = \
                    ECTNestedDict[file]['Presentation'][speakers[-1]][count - 1] + ' ' + tag.text
                except IndexError:
                    ECTNestedDict[file]['Presentation'][speakers[-1]].append(tag.text)
                # uncommment this if each para counts as spoken different times
                # ECTNestedDict[file]['Presentation'][speakers[-1]].append(tag.text)
    # print(ECTNestedDict[file]['Presentation'])

    ECTNestedDict[file]['Questionnaire'] = {}
    speakers = []
    strong_flag = 0
    for tag in tags:
        if tag.find('strong') != None:
            heading = " ".join(tag.find('strong').text.split())
            if heading == 'Question-and-Answer Session':
                strong_flag = 1
                continue
            else:
                if strong_flag == 1:
                    if 'Q - ' in heading:
                        heading = heading.replace('Q - ', '')
                    speakers.append(heading)
                    speaker_number = len(speakers) - 1
                    ECTNestedDict[file]['Questionnaire'][speaker_number] = {}
                    ECTNestedDict[file]['Questionnaire'][speaker_number]['Speaker'] = speakers[-1]
                    ECTNestedDict[file]['Questionnaire'][speaker_number]['Remark'] = ''
                    continue
                else:
                    continue
        else:
            if strong_flag == 1 and len(speakers) != 0:
                speaker_number = len(speakers) - 1
                ECTNestedDict[file]['Questionnaire'][speaker_number]['Remark'] = \
                ECTNestedDict[file]['Questionnaire'][speaker_number]['Remark'] + ' ' + tag.text

    # Add extracted components to the Text dictionary
    TextDict[file] = ''
    TextDict[file] = TextDict[file] + ECTNestedDict[file]['Date'] + ' '
    for participant in ECTNestedDict[file]['Participants']:
        TextDict[file] = TextDict[file] + participant + ' '
    for presenter, presentation in ECTNestedDict[file]['Presentation'].items():
        TextDict[file] = TextDict[file] + presenter + ' ' + ' '.join(presentation) + ' '
    for speaker_number, qna in ECTNestedDict[file]['Questionnaire'].items():
        TextDict[file] = TextDict[file] + qna['Speaker'] + ' ' + qna['Remark'] + ' '

    #print('Converted HTML to Text successfully for: ' + file)

#print(TextDict)
#print('Building InvertedPositionalIndex')
InvertedPositionalIndex = {}

for file in files:
    content = TextDict[file]
    #remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    content = content.translate(translator)

    tokens = word_tokenize(content)

    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = lemmatized_tokens

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    punctuation_removed = [stop_word.translate(translator) for stop_word in stop_words]
    lemmatized_stopwords = [lemmatizer.lemmatize(stop_word) for stop_word in punctuation_removed]
    filtered_tokens = [token for token in tokens if not token in lemmatized_stopwords]
    # print(filtered_tokens)

    lemmatized_tokens = set(filtered_tokens)

    for lemmatized_token in lemmatized_tokens:
        try:
            InvertedPositionalIndex[lemmatized_token]['df'] = InvertedPositionalIndex[lemmatized_token]['df'] + 1
            InvertedPositionalIndex[lemmatized_token]['tf'].append((file, filtered_tokens.count(lemmatized_token)))
        except KeyError:
            InvertedPositionalIndex[lemmatized_token] = {}
            InvertedPositionalIndex[lemmatized_token]['df'] = 1
            InvertedPositionalIndex[lemmatized_token]['tf'] = []
            InvertedPositionalIndex[lemmatized_token]['tf'].append((file, filtered_tokens.count(lemmatized_token)))

#print(InvertedPositionalIndex)
N = 1000

InvertedPositionalIndex1 = {}
for key in InvertedPositionalIndex:
    idf = math.log10(N/InvertedPositionalIndex[key]['df'])
    InvertedPositionalIndex1[(key, idf)] = []
    for entry in InvertedPositionalIndex[key]['tf']:
        tf = math.log10(1 + entry[1])
        InvertedPositionalIndex1[(key, idf)].append((entry[0], tf))

#print(InvertedPositionalIndex1)

#print('Building ChampionListLocal')
ChampionListLocal = {}
for key in InvertedPositionalIndex1:
    # take second element for sort
    def takeSecond(elem):
        return elem[1]
    list = InvertedPositionalIndex1[key]
    # sort list with key
    list.sort(key = takeSecond, reverse = True)
    ChampionListLocal[key[0]] = []
    if len(list) < 51:
        for entry in list:
            ChampionListLocal[key[0]].append(entry[0])
    else:
        for i in range(50):
            ChampionListLocal[key[0]].append(list[i][0])
#print(ChampionListLocal)

#print('Building ChampionListGlobal')
path = r'../Dataset/StaticQualityScore.pkl'
static_quality_scores = pickle.load(open(path, 'rb'))

ChampionListGlobal = {}
for key in InvertedPositionalIndex1:
    # take second element for sort
    def takeSecond(elem):
        return elem[1]
    idf = key[1]
    list = InvertedPositionalIndex1[key]
    updated_list = []
    for item in list:
        tf = item[1]
        document_index = int(item[0].replace('.html',''))
        static_quality_score = static_quality_scores[document_index]
        score = static_quality_score + idf*tf
        updated_list.append((item[0], score))

    # sort list with key
    updated_list.sort(key = takeSecond, reverse = True)
    ChampionListGlobal[key[0]] = []
    if len(list) < 51:
        for entry in updated_list:
            ChampionListGlobal[key[0]].append(entry[0])
    else:
        for i in range(50):
            ChampionListGlobal[key[0]].append(updated_list[i][0])
#print(ChampionListGlobal)

# print('Creating Document Vectors')
normalized_document_vectors = {}

# creating an empty 2d array
normalized_document_array = np.empty((0, len(InvertedPositionalIndex1)), float)

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

    normalized_document_array = np.append(normalized_document_array, normalized_document_vector, axis=0)

# print(normalized_document_array)
# print('Answering Queries')

def query_results(query):
    content = query

    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    content = content.translate(translator)

    tokens = word_tokenize(content)
    #print('here-2')
    # lower case
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = lemmatized_tokens

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    punctuation_removed = [stop_word.translate(translator) for stop_word in stop_words]
    lemmatized_stopwords = [lemmatizer.lemmatize(stop_word) for stop_word in punctuation_removed]
    filtered_tokens = [token for token in tokens if not token in lemmatized_stopwords]
    lemmatized_tokens = set(filtered_tokens)

    query_vector = np.zeros((1, len(InvertedPositionalIndex1)), float)
    #print('here-1')
    #print(lemmatized_tokens)
    counter = 0
    for key in InvertedPositionalIndex1:
        flag = 0
        for lemmatized_token in lemmatized_tokens:
            if key[0] == lemmatized_token:
                flag = 1
                break
        if flag == 1:
            query_vector[0, counter] = key[1]
        else:
            query_vector[0, counter] = 0
        counter = counter + 1

    normalizing_factor = 0
    temp = np.square(query_vector)
    normalizing_factor = np.sum(temp)
    normalizing_factor = math.sqrt(normalizing_factor)
    if normalizing_factor != 0:
        normalized_query_vector = np.divide(query_vector, normalizing_factor)
    else:
        normalized_query_vector = query_vector

    #print('here0')
    # tf_idf_score(Q, d)
    tf_idf_score = {}
    counter = 0
    for file in files:
        #print('here3')
        score = np.sum(np.multiply(normalized_query_vector, normalized_document_array[counter,:]))
        tf_idf_score[file] = score
        counter = counter + 1
        #print('here4')

    #print(tf_idf_score)
    k = Counter(tf_idf_score)
    #print('here5')

    # Finding 10 highest values
    high = k.most_common(10)
    #print('here6')
    response = ''
    for i in high:
        response = response + '<' + str(i[0]) + ', ' + str(i[1]) + '>, '
    response = response[:-2]

    # Local_Champion_List_Score(Q, d)
    Local_Champion_List_Score = {}
    docs = []
    for lemmatized_token in lemmatized_tokens:
        try:
            for doc in ChampionListLocal[lemmatized_token]:
                docs.append(doc)
        except:
            print()
            # print('Query term not found in ChampionListLocal')
    docs = set(docs)
    for file in docs:
        index = files.index(file)
        score = np.sum(np.multiply(normalized_query_vector, normalized_document_array[index,:]))
        Local_Champion_List_Score[file] = score

    #print(Local_Champion_List_Score)
    k = Counter(Local_Champion_List_Score)

    # Finding 10 highest values
    high = k.most_common(10)
    response = response + '\n'
    for i in high:
        response = response + '<' + str(i[0]) + ', ' + str(i[1]) + '>, '
    response = response[:-2]

    # Global_Champion_List_Score(Q, d)
    Global_Champion_List_Score = {}
    docs = []
    for lemmatized_token in lemmatized_tokens:
        try:
            for doc in ChampionListGlobal[lemmatized_token]:
                docs.append(doc)
        except:
            print()
            #print('Query term not found in ChampionListGlobal')
    docs = set(docs)
    for file in docs:
        index = files.index(file)
        score = np.sum(np.multiply(normalized_query_vector, normalized_document_array[index,:]))
        Global_Champion_List_Score[file] = score

    #print(Global_Champion_List_Score)
    k = Counter(Global_Champion_List_Score)

    # Finding 10 highest values
    high = k.most_common(10)
    response = response + '\n'
    for i in high:
        response = response + '<' + str(i[0]) + ', ' + str(i[1]) + '>, '
    response = response[:-2]

    path = r'../Dataset/Leaders.pkl'
    leaders = pickle.load(open(path, 'rb'))
    leaders = [str(i) + '.html' for i in leaders]
    Leader_Score = {}

    for file in leaders:
        index = files.index(file)
        score = np.sum(np.multiply(normalized_query_vector, normalized_document_array[index, :]))
        Leader_Score[file] = score

    #print(Leader_Score)
    k = Counter(Leader_Score)

    # Finding 10 highest values
    high = k.most_common(1)
    for i in high:
        leader = str(i[0])

    pairwise_scores = {}
    cluster = []
    cluster.append(leader)
    for file in files:
        if file not in leaders:
            pairwise_scores[file] = {}
            index = files.index(file)
            for file1 in leaders:
                index1 = files.index(file1)
                score = np.sum(np.multiply(normalized_document_array[index, :], normalized_document_array[index1, :]))
                pairwise_scores[file][file1] = score

            k = Counter(pairwise_scores[file])

            # Finding 10 highest values
            high = k.most_common(1)
            flag1 = 0
            for i in high:
                if leader == str(i[0]):
                    flag1 = 1
                    break
            if flag1 == 1:
                 cluster.append(file)

    # Cluster_Pruning_Score(Q, d)
    Cluster_Pruning_Score = {}

    for file in cluster:
        index = files.index(file)
        score = np.sum(np.multiply(normalized_query_vector, normalized_document_array[index, :]))
        Cluster_Pruning_Score[file] = score

    #print(Cluster_Pruning_Score)
    k = Counter(Cluster_Pruning_Score)

    # Finding 10 highest values
    high = k.most_common(10)
    response = response + '\n'
    for i in high:
        response = response + '<' + str(i[0]) + ', ' + str(i[1]) + '>, '
    response = response[:-2]
    #print(response)
    return response

def iterate_over_queries(file):
    path = r'./%s' % (file)
    f = open(path, mode='r', encoding='utf-8')
    Lines = f.readlines()
    f.close()
    #print('here-3')
    result = ''
    for line in Lines:
        result = result + line.strip() + '\n' + query_results(line.strip()) + '\n\n'

    path = r'./RESULTS2_17EC10060.txt'
    f = open(path, mode='w', encoding='utf-8')
    f.write(result)
    f.close()

query_file = sys.argv[1]
#query_file = 'query.txt'
iterate_over_queries(query_file)

# print('Completed task')
