import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from os.path import isfile, join
import string
import json

#read HTML file
files = [f for f in listdir('./ECTText') if isfile(join('./ECTText', f))]
inverted_positional_index = {}

print('Task 3: Building Index')
for file in files:
    path = r'./ECTText/%s' %(file)
    f = open(path, mode='r', encoding='utf-8')
    content = f.read()
    f.close()

    #remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    content = content.translate(translator)

    tokens = word_tokenize(content)
    #tokens = [token.lower() for token in tokens if token.isalpha()]
    #print(tokens)

    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = lemmatized_tokens

    #Note that it is 0 based positioning
    position_details = {}
    position = 0
    for token in tokens:
        try:
            position_details[token].append(position)
        except KeyError:
            position_details[token] = []
            position_details[token].append(position)
        position = position + 1
    #print(position_details)

    #sort the positional index
    sorted_keys = sorted(position_details)
    temp = {}
    for key in sorted_keys:
        temp[key] = position_details[key]
    position_details = temp
    #print(position_details)

    #removing stopwords
    stop_words = set(stopwords.words('english'))
    punctuation_removed = [stop_word.translate(translator) for stop_word in stop_words]
    lemmatized_stopwords = [lemmatizer.lemmatize(stop_word) for stop_word in punctuation_removed]
    filtered_tokens = [token for token in tokens if not token in lemmatized_stopwords]
    #print(filtered_tokens)

    lemmatized_tokens = set(filtered_tokens)
    #lemmatizer = WordNetLemmatizer()
    #lemmatized_tokens = [lemmatizer.lemmatize(filtered_token) for filtered_token in filtered_tokens]
    #print(lemmatized_tokens)

    for lemmatized_token in lemmatized_tokens:
        try:
            inverted_positional_index[lemmatized_token].append([file, position_details[lemmatized_token]])
        except KeyError:
            inverted_positional_index[lemmatized_token] = []
            inverted_positional_index[lemmatized_token].append([file, position_details[lemmatized_token]])
    print('done everything for: ' + file)

#sort the inverted positional index
sorted_keys = sorted(inverted_positional_index)
temp = {}
for key in sorted_keys:
    temp[key] = inverted_positional_index[key]
inverted_positional_index = temp

#save inverted positional index into json file, so that it can be used in the next part
with open("inverted_positional_index.json", "w") as outfile:
    json.dump(inverted_positional_index, outfile)
#print(inverted_positional_index)

print('Completed Task 3')

