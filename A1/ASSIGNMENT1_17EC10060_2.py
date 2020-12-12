from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
import bs4
import json

#read HTML file
files = [f for f in listdir('./ECT') if isfile(join('./ECT', f))]
ECTNestedDict = {}
file_number = 0
print('Task 2: Building corpus')
for file in files:
    file_number = file_number + 1
    path = r'./ECT/%s' %(file)
    f = open(path, mode='r', encoding='utf-8')
    content = f.read()
    f.close()

    ECTNestedDict[file] = {}
    ECTNestedDict[file]['Date'] = ''

    soup = BeautifulSoup(content, features = 'html.parser')
    tags = soup.find_all('p')
    #tags = soup.find_all('p', attrs={'class' : 'p p1'})

    if len(tags) == 0:
        continue

    #print(file)
    text = []
    for x in tags[0]:
        if isinstance(x, bs4.element.NavigableString):
            text.append(x.strip())
    text = ' '.join(text)
    text = ' '.join(text.split('  '))
    text = text.split(' ')
    datentime = ' '.join(text[-6:-3])
    #print(datentime)
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
    #print(participants)

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
                #commment this if each para counts as spoken different times
                count = speakers.count(speakers[-1])
                try:
                    ECTNestedDict[file]['Presentation'][speakers[-1]][count - 1] = ECTNestedDict[file]['Presentation'][speakers[-1]][count - 1] + ' ' + tag.text
                except IndexError:
                    ECTNestedDict[file]['Presentation'][speakers[-1]].append(tag.text)
                #uncommment this if each para counts as spoken different times
                #ECTNestedDict[file]['Presentation'][speakers[-1]].append(tag.text)
    #print(ECTNestedDict[file]['Presentation'])

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
                    speaker_number = len(speakers)-1
                    ECTNestedDict[file]['Questionnaire'][speaker_number] = {}
                    ECTNestedDict[file]['Questionnaire'][speaker_number]['Speaker'] = speakers[-1]
                    ECTNestedDict[file]['Questionnaire'][speaker_number]['Remark'] = ''
                    continue
                else:
                    continue
        else:
            if strong_flag == 1 and len(speakers) != 0:
                speaker_number = len(speakers)-1
                ECTNestedDict[file]['Questionnaire'][speaker_number]['Remark'] = ECTNestedDict[file]['Questionnaire'][speaker_number]['Remark'] + ' ' + tag.text

    #print(ECTNestedDict[file]['Questionnaire'])
    #print('\n')

    path = r'./ECTText/%s' % (str(file_number) + '.txt')
    f = open(path, mode='w', encoding='utf-8')

    f.write(ECTNestedDict[file]['Date'])
    f.write('\n\n')

    for participant in ECTNestedDict[file]['Participants']:
        f.write(participant)
        f.write('\n')
    f.write('\n')

    for presenter, presentation in ECTNestedDict[file]['Presentation'].items():
        f.write(presenter + '\n' + '\n'.join(presentation) + '\n\n')

    for speaker_number, qna in ECTNestedDict[file]['Questionnaire'].items():
        f.write(qna['Speaker'] + '\n' + qna['Remark'] + '\n\n')

    f.close()

    # path = r'./ECTNestedDict/%s' % ('File' + str(file_number) + '.txt')
    # f = open(path, mode='w', encoding='utf-8')
    # f.write(str(ECTNestedDict))
    # f.close()
    print('done everything for: ' + file)

with open(r'./ECTNestedDict.json', "w") as f:
    json.dump(ECTNestedDict, f)
print('Completed Task 2')


