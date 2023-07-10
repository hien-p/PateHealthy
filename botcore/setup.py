from langchain.llms import VertexAI
import os
from dotenv import load_dotenv
import json
import streamlit as st

def load_my_env():
    env_path = os.path.dirname(__file__)
    load_dotenv(f'{env_path}/../.google/env')

# Trace

def enable_tracing(session:str='test-deploy') -> bool:
    load_my_env()
    lang_key = st.secrets['LANGCHAIN']
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
    os.environ["LANGCHAIN_API_KEY"] = lang_key
    os.environ["LANGCHAIN_SESSION"] = session
    print(f"Enable tracing at {session}")
    return True

def trace_palm2(model_name:str = 'text-bison', max_tokens:int = 280, session:str="test-deploy") -> VertexAI:
    enable_tracing(session)
    #service_json_path = f'{os.path.dirname(__file__)}/../.google/service_account.json'
    service_json_path = f"{os.path.dirname(__file__)}/service.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_json_path
    model = VertexAI(model_name=model_name, max_output_tokens=max_tokens,verbose=True, temperature=0.0)
    print("Vertex AI Palm 2 ready")
    return model
    
def build_service(data_str):
    data = json.loads(data_str)
    with open(f'{os.path.dirname(__file__)}/service.json', 'w') as f:
        json.dump(data, f)

