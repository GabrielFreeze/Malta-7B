import os
import torch
from typing import List
from random import randint
from docarray import DocList
from threading import Thread
from datetime import datetime
from vectordb import HNSWVectorDB
from utils import EmbDoc, augment_date
from auto_gptq import exllama_set_max_input_length
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, \
                         StoppingCriteriaList, TextIteratorStreamer


class GlobalBackend:
    def __init__(self, model_id:str, revision:str='main',
                       hf_token:str='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',
                       exllama:bool=True):
        
        self.default_from_yr = 2024
        self.default_to_yr   = 2024
        self.default_src_thresh = 0.68

        self.device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")

        #Load Retrieval Model & Related
        self.retrieval_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",trust_remote_code=True,device='cpu')

        self.model     = AutoModelForCausalLM.from_pretrained(model_id,device_map=self.device,token=hf_token,
                                                              offload_folder="offload", revision=revision)
        
        self.tokenizer = AutoTokenizer       .from_pretrained(model_id,device_map=self.device,token=hf_token, use_fast=True,
                                                              torch_dtype=torch.float32,offload_folder="offload")
        

        if exllama:
            self.model = exllama_set_max_input_length(self.model, max_input_length=100_000)
                  
class SessionBackend:
    def __init__(self, vectorDB_name:str="vector_db"):
        #Load device
        self.device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")

        self.stop_generation = False
        self.generate_kwargs = dict() # Holds input+hyperparameters to invoke model generation
        self.streamer = None
        self.system_prompt = ""
        with open("chat_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        
        self.previous_documents = ""
        self.retrieved_EmbDocs:List[EmbDoc] = []

        self.vector_db_manager = VectorDB_Manager(vectorDB_name=vectorDB_name)
        self.vector_db_manager.update_range(2024,2024)
        self.url_sources = set() #This will contain current extracted sources
        self.source_header = """<div style="text-align: center; font-size: 24px;">Sources</div>"""        

        return
    
    def update_range(self,from_yr,to_yr):
        return self.vector_db_manager.update_range(from_yr,to_yr)

    def format_sources(self, source:EmbDoc) -> str:
        
        return f"Title: {source.title}\n"      + \
               f"Date: {source.date}\n"        + \
               f"URL: {source.url}\n"          + \
               f"Author: {source.author}\n"    + \
               f"News Article: {source.body}\n\n"

    def get_context(self,history,retrieval_model,k:int=20):

        #Extract the user-inputted prompt
        prompt_query = history[-1][0]

        date = augment_date(datetime.today().strftime('%d-%m-%Y'))

        retrieval_query = f"Date: {date}\nsearch_query:{history[-1][0]}"

        ##OPTIONAL:Include AI response in retrieval query
        # for i,(user,ai) in enumerate(history[-2:]):
            
        #     if i == len(history):
        #         ai = ""
        #         user = 'search_query: '+ user
            
        #     retrieval_query += f"{user}\n{ai}\n"

        #Encode Retrieval Query
        q_emb = retrieval_model.encode(retrieval_query)
        results   = self.vector_db_manager.search(q_emb,limit=k)
        # results = vector_db.search(inputs=DocList[EmbDoc]([EmbDoc(embedding=q_emb)]), limit=k)
        #                              #Hyperparameter k: amount of articles to retrieve     ^^

        
        #Extract meta-data from retrieved items
        context_str = ""
        urls = []
        print('\n\DOCS:\n')

        #Format top k relevant results
        for i,m in enumerate(results[::-1]):
            
            score = (self.vector_db_manager
                         .cos(rtrv_emb := torch.tensor(m.embedding, device=self.device),
                              torch.tensor(q_emb, device=self.device))
                         .detach()
                         .item())

            print(f'{m.date} - {m.title} - {round(score,2)}')
            if score < 0.55: #This variable should really be a hyper-parameter
                print("======!REMOVE!======\n")
                
                continue
            
            context_str += self.format_sources(m)
            
            self.retrieved_EmbDocs.append(m)
        print('\n\n')
        
        return context_str
    
    def format_input(self,history,retrieval_model):

        #Augment latest user query with related news articles
        #Also pass top 3 retrieved sources to help with relevance.
        current_documents = self.previous_documents + self.get_context(history,retrieval_model)

        # chat_history = ""
        # last=len(history)-1
        # for i,(user_msg,ai_msg) in enumerate(history):
            
        #     #Latest Queries are prepended with `New`
        #     chat_history += f"{['New ',''][i==last]}Query: {user_msg}\n"
        #     chat_history += f"Assistant: {ai_msg}\n" if i != last else "Assistant: "
        
        # date = augment_date(datetime.today().strftime('%d-%m-%Y'))

        #NOTE: I am experimenting with putting all related articles at the start, then the chat history at the end.
        # Maybe it will help with the AI keeping on topic/conversation.
        #UPDATE: This seems to greatly improve system performance.
        # input_sequence = f'<s>[INST]{self.system_prompt}\nNEWS ARTICLES:{current_documents}\n{chat_history}[/INST]'

        messages = []

        last=len(history)-1
        for i,(user_msg,ai_msg) in enumerate(history):

            if i == 0:
                messages.append({"role":"user","content":f"{self.system_prompt}\nNews Articles:\n{current_documents}\nQuery: {user_msg}"})
            else:
                messages.append({"role":"user","content":user_msg})

            if i != last:
                messages.append({"role":"assistant","content":ai_msg})

        return messages

    def prepare_streamer(self, history, g_bk:GlobalBackend):
        
        print(self.previous_documents)

        messages = self.format_input(history,g_bk.retrieval_model)

        self.streamer = TextIteratorStreamer(g_bk.tokenizer, timeout=120,skip_prompt=True, skip_special_tokens=False) 

        # Initialise Model
        input_tokens = g_bk.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        input_kwarg = {"input_ids":input_tokens,
                       "attention_mask":torch.ones_like(input_tokens,device=self.device)}
        self.generate_kwargs = dict(input_kwarg,streamer=self.streamer,max_new_tokens=512,do_sample=True,
                               top_p=0.90,temperature=0.4,num_beams=1,
                               stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
        
        return

    def stream_response(self,history,g_bk:GlobalBackend):

        #Invoke Model
        t = Thread(target=g_bk.model.generate, kwargs=self.generate_kwargs)
        t.start()

        #Stream response
        history[-1][1] = ""
        for i,token in enumerate(self.streamer):
            # tkn_id = self.tokenizer.convert_tokens_to_ids(token[:-1])
            # print(f'({token},{tkn_id})',end='')
            if self.stop_generation:
                return ""
            
            history[-1][1] += token
            yield history

    def display_sources(self, thresh):
        return self.source_header + \
               '<div style="font-size: 15px;">'+ \
               "<br>".join([f'<a href="{url}" target=”_blank”>{title}</a>'
                            for url,title,score in self.url_sources if score >= thresh]) + \
               '</div>'

    def update_sources(self,history,thresh,
                       retrieval_model:SentenceTransformer):

        new_url_sources = []

        #Calculate cosine similarity between last response and documents
        #..to find out which documents the AI actually used
        rsp_emb = torch.tensor(retrieval_model.encode(history[-1][1]), device=self.device)

        for m in self.retrieved_EmbDocs:
            similarity = (self.vector_db_manager
                              .cos(torch.tensor(m.embedding, device=self.device),
                                   rsp_emb))

            new_url_sources.append((m,similarity))

            
        #Filter previously retrieved documents with only the top 3 similair sources.
        to_add = sorted(new_url_sources,reverse=True,key=lambda x: x[1].detach().cpu())[:3]
        try:
            self.previous_documents += self.format_sources(to_add[0][0])
            self.previous_documents += self.format_sources(to_add[1][0])
            self.previous_documents += self.format_sources(to_add[2][0])
        except:
            pass

        #Format url_sources in set and remove similarity scores
        new_url_sources = set([(m.url,m.title,score) for m,score  in new_url_sources])

        #This is so fucking dirty bro        vv
        self.url_sources |= {(f"{randint(0,10e10)}","\n",
                              torch.tensor(1.0,device=self.device))} | new_url_sources 

        #Update sourceBox with new sources based on the threshold
        return self.display_sources(thresh)

    def reset(self):
        self.previous_documents = ""
        self.retrieved_EmbDocs = []
        self.url_sources = set()

        with open("chat_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        return

class VectorDB_Manager:
    def __init__(self,vectorDB_name:str='vector_db', data_path:str=os.path.join('..','data'),
                 from_yr:int=2023,to_yr:int=2024):
        
        self.device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

        self.vectorDB_name = vectorDB_name
        self.data_path = data_path
        
        self.cos = torch.nn.CosineSimilarity(dim=0) #To calculate cosine similarity between embeddings     
        self.from_yr = from_yr
        self.to_yr   = to_yr

        self.MIN_YR = 2018
        self.MAX_YR = 2024

        self.yr_range:List[int] = list(range(from_yr ,to_yr+1))

    def search(self,q_emb,limit:int=40):
        
        #NOTE: vector_db buckets are loaded for each query because accessing the vector from a different thread don't work
        vector_db = [HNSWVectorDB[EmbDoc](workspace=os.path.join(self.data_path, f"{self.vectorDB_name}_{yr}"))
                                                                           for yr in self.yr_range]

        #Get a list of the top relevant documents from each vector_db bucket
        retrieved_EmbDocs = [v_db.search(inputs=DocList[EmbDoc]([EmbDoc(embedding=q_emb)]),
                                      limit=limit)[0].matches
                          for v_db in vector_db]
        
        #Flatten list across different buckets
        retrieved_docs:List = []
        for embDoc in retrieved_EmbDocs:
            retrieved_docs += [m for m in embDoc]


        #Return top 40 most relevant documents        
        return sorted(retrieved_docs, key=lambda m: self.cos(torch.tensor(m.embedding, device=self.device),
                                                             torch.tensor(q_emb      , device=self.device)))[:40]

    def update_range(self,from_yr,to_yr):        
        
        #Update new year ranges
        self.from_yr = from_yr
        self.to_yr   = to_yr

        #Get indices
        self.yr_range = list(range(from_yr ,to_yr+1))

        return

        
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:       

        for stop_id in []:
            if input_ids[0][-1] == stop_id:
                return True
        return False