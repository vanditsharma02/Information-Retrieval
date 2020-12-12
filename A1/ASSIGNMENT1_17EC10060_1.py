import requests
import time
from bs4 import BeautifulSoup
import random

def grab_page(url, page_num, item_num, headers):
    print("attempting to grab page: " + url)
    try:
        page = requests.get(url, headers = headers)
        page_html = page.text
        soup = BeautifulSoup(page_html, 'html.parser')
        content = soup.find(id="a-body")
        #print(content)
        if type(content) == "NoneType" or content is None:
            print("skipping this link, no content here")
            return
        else:
            filename = 'File_' + str(page_num) + '_'  +  str(item_num)
            #filename = get_ticker(mtext) + "_" + get_date(mtext)
            file = open('./ECT/' + filename.lower() + ".html", 'w', encoding = 'utf-8')
            file.write(str(content))
            file.close
            print(filename.lower()+ " sucessfully saved")
    except:
        return

def process_list_page(i):

    user_list = [
        # Chrome
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        # Firefox
        'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
        'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
    ]
    headers = {"User-Agent": random.choice(user_list)}

    page_num = i
    origin_page = "https://seekingalpha.com/earnings/earnings-call-transcripts" + "/" + str(i)
    print("getting page " + origin_page)
    page = requests.get(origin_page)
    page_html = page.text
    soup = BeautifulSoup(page_html, 'html.parser')
    alist = soup.find_all("li", {'class': 'list-group-item article'})

    for i in range(0,len(alist)):
        url_ending = alist[i].find_all("a")[0].attrs['href']
        url = "https://seekingalpha.com" + url_ending
        grab_page(url, page_num, i, headers)
        random_sleep = 5 / random.randrange(1, 5)
        time.sleep(random_sleep)

for i in range(1,500):
    random_sleep = 5 / random.randrange(1, 5)
    time.sleep(random_sleep)
    process_list_page(i)
