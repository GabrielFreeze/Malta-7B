from docarray import BaseDoc
from docarray.typing import NdArray
from datetime import datetime as dt

class color:
    WHITE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ESC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class EmbDoc(BaseDoc):
    title: str = ""
    author: str = ""
    date: str = ""
    caption: str = ""
    url: str = ""
    body: str = ""
    embedding: NdArray[768,] #<-- Needs to match shape of stored embeddings

def augment_date(date:str):
        
    #31 suffices that follow the day
    sfx = ['st','nd','rd','th','th','th','th',
           'th','th','th','th','th','th','th',
           'th','th','th','th','th','th','st',
           'nd','rd','th','th','th','th','th',
           'th','th','st']

    old_date = date

    try:
        date = dt.strptime(date, '%d-%m-%Y') if date != 'NULL' else 'NULL'
    except: date = dt.strptime(date, '%Y-%m-%d') if date != 'NULL' else 'NULL'
    
    date = date.strftime(f'%B %d{sfx[date.day-1]} %Y')\
                         if date != 'NULL' else 'NULL'

    
    date += ' ('+old_date+')' #Add numerical format in brackets
    
    return date