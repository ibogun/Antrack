__author__ = 'Ivan'

import httplib2
from BeautifulSoup import BeautifulSoup, SoupStrainer



if __name__ == "__main__":


    url="http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html"

    http = httplib2.Http()
    status, response = http.request(url)

    print response

    for link in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):

        print link
        if link.has_attr('href'):
            print link['href']