from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from InstructorEmbedding import INSTRUCTOR

class BaseSemanticRetrieval:

    def __init__(self,
                 model_name:str):
        
        self.model_name = model_name
        self.model = None

    def embed_documents(self, texts:list):
        return self.model.encode(texts)

    def embed_query(self, query:str):
        raise NotImplementedError
    
    def embed_corpus(self, documents:list):
        raise NotImplementedError
    
    def retrieve_relevant(self,
                          query_emb,
                          document_embeddings):
        
        # print("query embeddings : ", query_emb)
        # print("doc embeddings : ", document_embeddings)

        relevant_docs = semantic_search(query_embeddings=query_emb,
                                        corpus_embeddings=document_embeddings,
                                        top_k=10)[0]
        
        return relevant_docs
    
class SemanticRetrieval(BaseSemanticRetrieval):

    def __init__(self,
                 model_name: str = "multi-qa-mpnet-base-dot-v1"):
        
        super().__init__(model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, query: str):
        return self.embed_documents(texts=query)
    
    def embed_corpus(self, documents: list):
        return self.embed_documents(texts=documents)
    
class InstructRetrieval(BaseSemanticRetrieval):

    def __init__(self,
                 model_name: str = "hkunlp/instructor-large"):

        super().__init__(model_name)
        self.model = INSTRUCTOR(self.model_name)

    def embed_query(self, query: str):

        instruction = "Represent the query for answering questions : "
        query = [instruction, query]

        return self.embed_documents(texts=query)

    def embed_corpus(self,
                        documents: list):
        
        if not isinstance(documents, list):
            documents = [documents]

        instruction = "Represent the statement : "
        documents = [[instruction, doc] for doc in documents]

        return self.embed_documents(texts=documents)