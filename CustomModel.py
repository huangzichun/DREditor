from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict, Union, Tuple
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm

class YourCustomDEModel:
    def __init__(self, model_path=None, max_seq_length=512,device ='cpu',sep='[sep]',**kwargs):
        self.device = device
        self.sep = sep

        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-multiset-base')
        self.doc_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.q_model_base = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-multiset-base')#todo
        self.doc_model_base = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

        self.q_model_base.to(self.device)
        self.doc_model_base.to(self.device)

        self.max_seq_length = max_seq_length

        self.q_model_linear = nn.Linear(self.q_model_base.config.hidden_size, self.q_model_base.config.hidden_size)
        self.doc_model_linear = nn.Linear(self.doc_model_base.config.hidden_size, self.doc_model_base.config.hidden_size)

        self.q_model_linear.load_state_dict(torch.load(model_path[0]))
        self.doc_model_linear.load_state_dict(torch.load(model_path[1]))

        self.q_model_linear.eval()
        self.doc_model_linear.eval()
        self.q_model_linear.to(self.device)
        self.doc_model_linear.to(self.device)

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        q = data_collect(queries)
        q_data = DataLoader(dataset=q, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            query_output = []
            data_iter = tqdm.tqdm(enumerate(q_data),
                                  desc="query enmbedding",
                                  total=len(q_data),
                                  bar_format="{l_bar}{r_bar}")
            for i, data in data_iter:
                query_token = self.q_tokenizer(data, padding="max_length", max_length=self.max_seq_length, truncation=True)
                query_token = {key: torch.tensor(value).to(self.q_model_base.device) for key, value in query_token.items()}
                query_output_tmp = self.q_model_base(**query_token).pooler_output.detach()
                query_output.extend(self.q_model_linear(query_output_tmp).cpu())

        return np.asarray([emb.numpy() for emb in query_output])

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]
        c = data_collect(sentences)
        c_data = DataLoader(dataset=c, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            doc_output = []
            data_iter = tqdm.tqdm(enumerate(c_data),
                                  desc="corpus enmbedding",
                                  total=len(c_data),
                                  bar_format="{l_bar}{r_bar}")
            for i, data in data_iter:
                doc_token = self.doc_tokenizer(data, padding="max_length", max_length=self.max_seq_length, truncation=True)
                doc_token = {key: torch.tensor(value).to(self.doc_model_base.device) for key, value in doc_token.items()}
                doc_output_tmp = self.doc_model_base(**doc_token).pooler_output.detach()
                doc_output.extend( self.doc_model_linear(doc_output_tmp).cpu())

        return np.asarray([emb.numpy() for emb in doc_output])

class data_collect(Dataset):
    def __init__(self, data):
        super(data_collect, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)