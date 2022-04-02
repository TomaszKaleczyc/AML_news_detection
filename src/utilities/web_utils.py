from bs4 import BeautifulSoup
from urllib.request import Request, urlopen


def get_webpage_body(link: str, user_agent: str = 'Mozilla/5.0') -> str:
    """
    Returns the text under a given webpage body
    """
    request = Request(link, headers={'User-Agent': user_agent})
    webpage=urlopen(request).read()
    soup = BeautifulSoup(webpage, features="html.parser")
    return soup.body.get_text()
