import urllib.request
from bs4 import BeautifulSoup

import requests

def get_page(url):
    """Scrapes a URL and returns the HTML source.
    Args:
        url (string): Fully qualified URL of a page.
    Returns:
        soup (string): HTML source of scraped page.
    """
    
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 
                         'html.parser', 
                         from_encoding=response.info().get_param('charset'))
    return soup



if __name__ == '__main__':
    page_name = 'https://www.kiabi.ru/'
    bsoup = get_page(page_name)

    ##################################
    