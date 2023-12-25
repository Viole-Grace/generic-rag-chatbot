#basic QA pipeline
import os
from uuid import uuid4

import re

import numpy as np
import pandas as pd

from utils.parsers import DocxParser, PDFParser
from retrieval.semantic_retrieval import SemanticRetrieval, InstructRetrieval
from llm_backends.api_models import GeminiLLM

class DocumentProcessor:

    def __init__(self):

        self.basic_document_store = None
        self.embedding_store = None

        self.embedding_model = None

        self.chunking_strategy = None

    def parse_documents(self,
                        document_paths):
        
        if not isinstance(document_paths, list):
            document_paths = [document_paths]

        parsed_documents = []
        
        for document in document_paths:

            parsed_document = ""

            try:
                extension = str(document).split(".")[-1]

                if extension in ["doc", "docx"]:
                    parser = DocxParser(document)
                if extension in ["pdf"]:
                    parser = PDFParser(document)

                parsed_document = parser.parse()
                parsed_document = parsed_document.replace("\t", " ").replace("\n", " ").strip()
                parsed_documents.append(parsed_document)

            except Exception as e:
                print(e)

        return parsed_documents

    def chunk_documents(self,
                        parsed_documents,
                        by_line:bool=False):
        
        chunks = []

        #chunk each line, and if the length of a line exceeds 400 words, break it and chunk it as two different lines
        if by_line:
            parsed_documents = [re.split(r'(?<=[.!?;])\s+', document)
                                for document in parsed_documents]
            parsed_documents = [arr for sublist in parsed_documents for arr in sublist]
        
        for document in parsed_documents:

            doc_len = len(document.split())
            chunk_id = str(uuid4())

            chunk = [{"doc" : document,
                     "id" : chunk_id,
                     "type" : "single"}]

            #around 400 words will be 512 tokens
            if doc_len > 400:

                max_chunks = doc_len//400
                max_chunks += 1
                
                multi_chunk_doc = [document.split()[idx*400 : (idx+1)*400]
                                   for idx in range(max_chunks)]
                
                multi_chunk_doc = [" ".join(doc).strip() for doc in multi_chunk_doc]

                print(f"len : {len(multi_chunk_doc)} | num tokens : {doc_len} | max_chunks : {max_chunks}")
                
                chunk = [{"doc" : doc, "id" : str(uuid4()), "type" : "multi_chunk"}
                         for doc in multi_chunk_doc]
        
            chunks.extend(chunk)

        return chunks

    def embed_documents(self,
                        chunked_documents):

        # embedding_model = InstructRetrieval()
        embedding_model = SemanticRetrieval()
        chunked_embeddings = {}

        # single_chunk_docs = [doc for doc in chunked_documents
        #                      if doc["type"] == "single"]
        # multi_chunk_docs = [doc for doc in chunked_documents
        #                     if doc["type"] == "multi_chunk"]
        
        # if len(single_chunk_docs) > 0:
        #     chunked_embeddings = embedding_model.embed_documents([doc["doc"] for doc in single_chunk_docs])
        #     chunked_embeddings = {single_chunk_docs[idx]["id"] : chunked_embeddings[idx]
        #                         for idx in range(len(single_chunk_docs))}

        # for mc_doc in multi_chunk_docs:

            # split_doc = mc_doc["doc"].split()
            # max_chunks = len(split_doc)//400
            # max_chunks += 1

            # chopped_up = [split_doc[idx*400 : (idx+1)*400]
            #                 for idx in range(max_chunks)]
            # chopped_up = [" ".join(_).strip() for _ in chopped_up]

            # print('chopped up doc : ',chopped_up)

            # chopped_up_emb = embedding_model.embed_documents(documents=split_doc)
            # mc_doc_emb = np.mean(chopped_up_emb, axis=0)

            # chunked_embeddings.update({mc_doc["id"] : mc_doc_emb})

        chunked_embeddings = embedding_model.embed_corpus([doc["doc"] for doc in chunked_documents])
        self.embedding_store = {chunked_documents[idx]["id"] : chunked_embeddings[idx]
                              for idx in range(len(chunked_documents))}
        self.embedding_store = [{"id" : corpus_id, "embedding" : embedding}
                                for corpus_id, embedding in self.embedding_store.items()]

        self.basic_document_store = chunked_documents
        self.embedding_model = embedding_model

        return chunked_embeddings

class DocumentQALLM:

    def __init__(self,
                 document_processor,
                 llm_backend=GeminiLLM()):
        
        self.document_processor = document_processor
        self.llm = llm_backend

        self.document_store = document_processor.basic_document_store
        self.embedding_store = document_processor.embedding_store
        self.embedding_model = document_processor.embedding_model

        self.embedding_store = pd.merge(pd.DataFrame(self.document_store),
                                        pd.DataFrame(self.embedding_store),
                                        on=["id"],
                                        how="left")

    def retrieve_correct_context(self,
                                 query):
        
        query_emb = self.embedding_model.embed_query(query)
        document_embeddings = self.embedding_store["embedding"].values
        document_embeddings = np.array([np.array(embedding) for embedding in document_embeddings]) #inefficient but vanilla -- will move to a vector store like pinecone or chroma for the next version
        
        context = self.embedding_model.retrieve_relevant(query_emb=query_emb,
                                                         document_embeddings=document_embeddings)
        

        _documents = self.embedding_store["doc"].tolist() #VERY inefficient having to get and lookup for every retrieval query -- will move to a vector store for the next version
        for item in context:
            corpus_idx = item['corpus_id']
            item["doc"] = _documents[corpus_idx]
        
        return context

    def query_with_llm(self,
                       query:str):
        
        context = self.retrieve_correct_context(query=query)

        print("query used : ", query)
        print("context used : ",context)

        #format the context to only have documents without the index and score
        llm_context = [relevant_document['doc'] for relevant_document
                       in context]

        answer = self.llm.prompt_llm(query=query, context=llm_context)

        return answer
