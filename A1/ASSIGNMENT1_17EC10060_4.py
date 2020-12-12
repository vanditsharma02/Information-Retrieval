import json
import sys

print('Task 4: Answering wildcard queries with single * symbol')

with open('inverted_positional_index.json') as f:
   inverted_positional_index = json.load(f)

def generate_permuterm(input):
    input = input + '$'
    combinations = []
    for i in range(len(input)):
        combinations.append(input[i:]+input[:i])
    return combinations

permuterm_index = {}
for key, values in inverted_positional_index.items():
    for permuterm in generate_permuterm(key):
        permuterm_index[permuterm] = inverted_positional_index[key]

def undo_permuterm(permuterm):
    d = permuterm.find('$')
    return permuterm[d + 1:] + permuterm[:d]

def query_results(query):
    query = query.lower()
    #response = "train:<25,1>,<25,19>,<29,1>,<29,6>; trail:<3,1>,<6,1>,<6,10>"
    response = ''
    if not '*' in query:
        lookup = query + '$'
        if lookup in permuterm_index:
            response = response + query + ':'
            for i in range(len(permuterm_index[lookup])):
                for j in range(len(permuterm_index[lookup][i][1])):
                    response = response + '<' + permuterm_index[lookup][i][0][:-4] + ',' + str(permuterm_index[lookup][i][1][j]) + '>,'
            response = response[:-1]
            print(response)
            #print(undo_permuterm(lookup) + ': ')
            #print(permuterm_index[lookup])
            return response
    elif query[-1] == '*':
        lookup = '$' + query[:-1]
        for key in permuterm_index:
            if lookup == key[0:len(lookup)]:
                response = response + ';' + undo_permuterm(key) + ':'
                for i in range(len(permuterm_index[key])):
                    for j in range(len(permuterm_index[key][i][1])):
                        response = response + '<' + permuterm_index[key][i][0][:-4] + ',' + str(
                            permuterm_index[key][i][1][j]) + '>,'
                response = response[:-1]
                #print(undo_permuterm(key) + ': ')
                #print(permuterm_index[key])
        response = response[1:]
        print(response)
        return response
    elif query[0] == '*':
        lookup = query[1:] + '$'
        for key in permuterm_index:
            if lookup == key[0:len(lookup)]:
                response = response + ';' + undo_permuterm(key) + ':'
                for i in range(len(permuterm_index[key])):
                    for j in range(len(permuterm_index[key][i][1])):
                        response = response + '<' + permuterm_index[key][i][0][:-4] + ',' + str(
                            permuterm_index[key][i][1][j]) + '>,'
                response = response[:-1]
                #print(undo_permuterm(key) + ': ')
                #print(permuterm_index[key])
        response = response[1:]
        print(response)
        return response
    else:
        star_position = query.find('*')
        X = query[0:star_position]
        Y = query[star_position+1:]
        lookup = Y + '$' + X
        for key in permuterm_index:
            if lookup == key[0:len(lookup)]:
                response = response + ';' + undo_permuterm(key) + ':'
                for i in range(len(permuterm_index[key])):
                    for j in range(len(permuterm_index[key][i][1])):
                        response = response + '<' + permuterm_index[key][i][0][:-4] + ',' + str(
                            permuterm_index[key][i][1][j]) + '>,'
                response = response[:-1]
                #print(undo_permuterm(key) + ': ')
                #print(permuterm_index[key])
        response = response[1:]
        print(response)
        return response

def iterate_over_queries(file):
    path = r'./%s' % (file)
    f = open(path, mode='r', encoding='utf-8')
    Lines = f.readlines()
    f.close()

    result = ''
    for line in Lines:
        result = result + query_results(line.strip()) + '\n'

    path = r'./RESULTS1_17EC10060.txt'
    f = open(path, mode='w', encoding='utf-8')
    f.write(result)
    f.close()

query_file = sys.argv[1]
#print(query_file)
iterate_over_queries(query_file)

print('Completed Task 4')




