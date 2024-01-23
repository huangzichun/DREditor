# DReditor: An Time-efficient Approach for Building a Domain-specific Dense Retrieval Model
> **Co-author**: Duanyu Feng (Sichuan University), Chen Huang (Sichuan University)

This is the code of our model **DReditor**, which is an computation-effective approach for domain-specific Dense Retrieval (DR) with training free.

## Paper
Coming soon

## Requirements
Our framework is based on the previous work, BEIR, which is a heterogeneous benchmark containing diverse IR tasks. Therefore, the requirements of our framework are the same as the BEIR. See more detail in 

> [beir-cellar/beir: A Heterogeneous Benchmark for Information
> Retrieval. Easy to use, evaluate your models across 15+ diverse IR
> datasets. (github.com)](https://github.com/beir-cellar/beir)

## DReditor
The core of our framework is an editor layer, which is a simple plug-in code. The code is in the file **Mylayereditor.py**. With this file, you can use BEIR to test any Dense Retrieval models supported by BEIR.

An example:

    from beir import util, LoggingHandler
    from beir.retrieval import models
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.search.sparse import SparseSearch
    from Mylayereditor import Mylayereditor
    from CustomModel import YourCustomDEModel
    import random
    import logging
    import pathlib, os

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### print debug information to stdout
    os.environ["CUDA_VISIBLE_DEVICES"]='5'
    #### Download scifact.zip dataset and unzip the dataset
    dataset = "scifact"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### the data we need to edit for the origin model
    corpus_edit, queries_edit, qrels_edit = GenericDataLoader(data_folder=data_path).load(split="train")

    #### the data we need to test
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### Load the dpr model and retrieve using cosine-similarity
    model = models.SentenceBERT(("facebook-dpr-question_encoder-multiset-base", "facebook-dpr-ctx_encoder-multiset-base", " [SEP] "))

    #### our plug-in for editing the model
    model_run = Mylayereditor(model, batch_size=16, calibrate_type="q")
	delta = model_run.layer_editor(corpus_t, queries_t, qrels_t, limits=500)

    #### Evaluate the model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    model_run.turn_on()
    retriever = EvaluateRetrieval(model_run, score_function="cos_sim") 
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


