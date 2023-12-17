import logging
import sys
import torch
from typing import Dict, List
from beir.util import cos_sim, dot_score
from beir.retrieval.search.dense import DenseRetrievalExactSearch
import random

logger = logging.getLogger(__name__)

class Mylayereditor(DenseRetrievalExactSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, calibrate_type="q", **kwargs):
        super(Mylayereditor, self).__init__(model, batch_size, corpus_chunk_size)
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.show_progress_bar = True  # TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}
        self._open_ROME = False
        self.delta = None
        self.calibrate_type = calibrate_type

    def layer_editor(self,
                     corpus: Dict[str, Dict[str, str]],
                     queries: Dict[str, str],
                     qrels: Dict[str, Dict[str, int]],
                     limits=-1,
                     lam=1.5*10**4):
        # Create embeddings for all queries using model.encode_queries()
        # Create embeddings for some useful corpus using model.encode_corpus()
        # Returns the change of the weight (emb_dim, target_dim)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())

        if limits != -1 and len(query_ids) > limits:
            query_ids = random.sample(query_ids, min(len(query_ids), limits))
        else:
            query_ids = query_ids

        queries = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor)#,device = 'cpu'

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_embeddings = torch.zeros_like(query_embeddings)

        for i, qid in enumerate(query_ids):
            corpus_id = list(qrels[qid].keys())
            corpus_label = torch.Tensor(list(qrels[qid].values()))

            sub_corpus = [corpus[cid] for cid in corpus_id]
            # print(sub_corpus)
            sub_corpus_embeddings = self.model.encode_corpus(
                sub_corpus,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor
                # device = 'cpu'
            )

            corpus_weight = (corpus_label / torch.sum(corpus_label)).to(sub_corpus_embeddings.device)

            corpus_embeddings[i, :] = corpus_weight @ sub_corpus_embeddings

        return self.Rome(query_embeddings, corpus_embeddings, lam)

    def Rome(self, k, v, lam=1.5*10**4):
        '''
        get the change of the weight
        :param k: the matrix of the new embedding (num, emb_dim)
        :param v: the matrix of the old embedding (num, emb_dim) and the target for the new embedding (num, target_dim)
        :param old_weight: the old weight of the Linear layer (target_dim, emb_dim)
        :return: the change of the weight (emb_dim, target_dim)
        '''
        old_emb = v
        new_emb = k
        target = v
        old_weight = torch.eye(v.size()[-1]).to(old_emb.device) #Todo add the last layer condition
        C0 = lam * torch.mean(old_emb.unsqueeze(-1) @ old_emb.unsqueeze(-2), dim=0)
        R = target - new_emb @ old_weight.T
        I = torch.eye(C0.shape[0]).to(old_emb.device)
        delta = R.T @ new_emb @ torch.linalg.solve(C0 + new_emb.T @ new_emb, I)
        self._open_ROME = True
        self.delta = delta.T
        self.manual_swatch = False

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: List[int],
               score_function: str,
               return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)

        if self.delta is not None and self._open_ROME and self.manual_swatch:
            print("editing...")
            query_embeddings = query_embeddings @ (self.delta.cuda() + torch.eye(self.delta.shape[0]).cuda())

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            #Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor = self.convert_to_tensor
            )

            if self.delta is not None and self._open_ROME and self.manual_swatch and self.calibrate_type == "qa":
                print("editing...")
                sub_corpus_embeddings = sub_corpus_embeddings @ (self.delta.cuda() + torch.eye(self.delta.shape[0]).cuda())

            #Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            #Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        self.results[query_id][corpus_id] = score

        return self.results

    def turn_on(self):
        self.manual_swatch = True

    def turn_off(self):
        self.manual_swatch = False