import os
import google.generativeai as genai

import json
import requests

gemini_api_key = os.environ["GEMINI_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

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

class GPTLLM(BaseModel):

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo"):
        
        super().__init__(model_name)
        self.llm = None

    def prompt_llm(self,
                   query,
                   context):
        
        messages = []

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}',
        }

        prompt = self.user_prompt.format(query)
        prompt += f"Use any of the context and data given here if useful and necessary in answering the query :\n{context}"

        system_prompt = [{"role" : "system", "content" : self.system_prompt}]
        user_prompt = [{"role" : "user", "content" : prompt}]

        messages.extend(system_prompt)
        messages.extend(user_prompt)

        data = {  
                    "model": self.model_name,  
                    "messages": messages,  
                    "temperature": 0  
        }

        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers,
                                 data=json.dumps(data))
        response_json = response.json()
        response_text = ""

        try:
            response_text = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(response_json)
            response_text = f"request could not be processed : {e}.\nPlease try later."

        return response_text