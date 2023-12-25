import os
import google.generativeai as genai

gemini_api_key = os.environ["GEMINI_API_KEY"]
# openai_api_key = os.environ["OPENAI_API_KEY"]

genai.configure(api_key=gemini_api_key)

class BaseModel:

    def __init__(self,
                 model_name:str):

        self.system_prompt = """You are an intelligent and helpful bot that helps answer queries.
                                You are a helpful, respectful and honest assistant. Always answer as 
                                helpfully as possible.

                                If a question does not make any sense, or is not factually coherent, explain 
                                why instead of answering something not correct. If you don't know the answer 
                                to a question, please don't share false information.

                                Your goal is to provide helpful and concise answers."""
        self.user_prompt = "query : {}\n"

        self.model_name = model_name
        self.llm = None

    def prompt_llm(self):
        raise NotImplementedError
    

class GeminiLLM(BaseModel):

    def __init__(self,
                 model_name: str = "models/gemini-pro"):
        
        super().__init__(model_name)
        self.llm = genai.GenerativeModel(self.model_name)

    def prompt_llm(self,
                   query,
                   context):
        
        prompt = self.system_prompt

        prompt += self.user_prompt.format(query)
        prompt += f"Use any of the context and data given here if useful and necessary in answering the query :\n{context}"

        res = self.llm.generate_content(prompt)
        return res.text
