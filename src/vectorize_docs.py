import os
import sys
import torch
import pandas as pd
from time import time
from tqdm import tqdm
from utils import EmbDoc, augment_date
from docarray import DocList
from typing import Union, List
from vectordb import HNSWVectorDB
from sentence_transformers import SentenceTransformer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

def pad_text(txt:str, pad_length:int):
    return txt.ljust(pad_length)
def repeat_text(txt:str,rep_length:int):
    s = ""
    l = len(txt)
    for i in range(rep_length):
        s += txt[i%l]
    return s
def chunk_text(txt:str, chunk_sz:int,
               sentence:bool=False):

    assert chunk_sz >= 0
    chunks   = []
    p_idx    = 0
    l = len(txt)

    stop_token = [' ','.'][sentence]
        
    while True:
        idx = p_idx+chunk_sz+1
        
        if l < idx:
            if p_idx-1 != l:
                chunks += [txt[p_idx:].ljust(chunk_sz,"☻")] 
            
            return chunks
    
        #Do not truncate mid-word.
        while idx<l and txt[idx]!=stop_token:
            idx+=1

        chunks.append(txt[p_idx:idx+sentence])
        p_idx=idx+1
def prepare_data(df:pd.DataFrame,chunk_sz:int=512) -> pd.DataFrame:
    pad = chunk_sz//20

    df['chunk'] = df['Body'].apply(lambda x: chunk_text(x,chunk_sz,True))
    df = df.explode('chunk').reset_index(drop=True)

    df['chunk'] = '\nText: '+df['chunk']
    for c in df.columns:
        if c not in ['chunk','Body']:
            df['chunk'] = c+': '+df[c]+'\n'+df['chunk']
    
    df['chunk'] = df['chunk'].apply(lambda x: x.ljust(l:=chunk_sz+pad,'☻')[:l])

    return df.reset_index(drop=True)

def prepare_data_2(df:pd.DataFrame,tokenizer:BertTokenizerFast, max_tkn_sz:int=8192) -> pd.DataFrame:

    #NOTE: As is, this function discards documents above `max_tkn_sz`.
    #TODO: Documents that exceed `max_tkn_sz` should have the `Body` chunked.


    df['chunk'] = '\nNews Article: '+df['Body']
    for c in df.columns:
        if c not in ['chunk','Body']:
            df['chunk'] = c+': '+df[c]+'\n'+df['chunk']
    
    
    #Load Retrieval System Prompt
    with open("retrieval_system_prompt.txt", "r") as f:
        system_prompt = f.read()   

    # /nomic-embed-text-v1/ must include this prefix for encoding documents
    #                                         vv  
    df['chunk'] = f' {system_prompt} search_document: ' + df['chunk']
    df['tkn_len'] = (df['chunk'].apply(tokenizer)
                                .apply(lambda tkns: len(tkns['input_ids'])))
    # df['Tokens'] = df['chunk'].apply(tokenizer)
    
    #Removes long documents
    df = df[df['tkn_len']<=max_tkn_sz]
        
    return df.reset_index(drop=True)

def main(data_path:Union[List[str],str]=os.path.join('..','data','data','tom_data.csv'),
         vector_db_name:str="vector_db"):

    #Load Model
    print(f"Device: {device}")
    # model = SentenceTransformer('BAAI/llm-embedder', device=device)  
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",
                                trust_remote_code=True,device=device)

    #Prepare the dataset
    s = time()
    print(f'Preparing Data: ',end='')

    if type(data_path) is str:
        data_path = [data_path]
    
    #Concatanate all csv files into one dataframe.
    #NOTE:This assumes that all csv files have the same column names
    df = (pd.concat([pd.read_csv(os.path.abspath(d)) for d in data_path],ignore_index=True)
            .reset_index(drop=True)
            .fillna('NULL'))
    
    #Ignoring newspaper name cause it is messing with retrieval
    if 'Newspaper' in df.columns:
        df = df.drop(['Newspaper'],axis=1)

    #TODO: I am augmenting/re-formatting the date here.
    #The date should already be augmented before vectorizing.
    df['Date'] = df['Date'].apply(augment_date)

    #Set all values to string
    df = df.astype(str)
    
    df = prepare_data_2(df,model.tokenizer)
    print(f'{round(time()-s,2)}s')

    #Prepare vectordb
    vector_db = HNSWVectorDB[EmbDoc](workspace=os.path.join('..','data',vector_db_name))

    #Encode
    s = time()
    print(f'Vectorizing Data: ',end='')
    # TODO: `model.encode` does not take already-tokenized documents.
    #So as it is now, tokenization is being performed twice (once for lenght, once to pass to encoding).
    #                       vv
    embs = model.encode(list(df['chunk']), device=device, batch_size=8,
                        show_progress_bar=True, normalize_embeddings=True)
    print(f'{round(time()-s,2)}s')


    #Add to vectordb
    s = time()
    print(f'Adding to VectorDB: ',end='')

    vector_list = list(df.apply(
        lambda x:EmbDoc(title=x['Title'],author =x['Author' ],
                        date =x['Date' ],caption=x['Caption'],
                        url  =x['URL'  ],body   =x['Body'   ],
                        embedding=embs[x.name]), axis=1))
    
    vector_db.index(inputs=DocList[EmbDoc](vector_list))

    print(f'{round(time()-s,2)}s')

    return 0

if __name__ == '__main__':
    
    p = os.path.join('..','data','data')

    # Valid Date Range
    assert sys.argv[1] <= sys.argv[2]
    
    for yr in range(sys.argv[1],sys.argv[2]+1):
        main(data_path=[os.path.join(p,f'data_{yr}.csv')],
            vector_db_name=f'vector_db_{yr}')