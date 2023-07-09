from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain import PromptTemplate, LLMChain
import json
from typing import List, Dict
from langchain.memory import ConversationBufferMemory


def load_sample_input():
    answers = ["No chronic diseases or food allergies.",
             "Likes fast food and quick-prep meals but wants to switch to a healthier eating regimen",
            "Weight loss and improving overall cardiovascular health.",
           "2 to 3 months."]
    
    questions = ["Can you share with me your history medical record?", "Can you tell me about your eating habits?",
            "What are your health goals?", "What is your timeline for achieving your goals?"]

    return questions, answers

def load_example_input():
    data_1 = {"medical_record": "No chronic diseases or food allergies.",
            "eating_habit": "Likes fast food and quick-prep meals but wants to switch to a healthier eating regimen",
            "health_goal": "Weight loss and improving overall cardiovascular health.",
            "timeline": "2 to 3 months."}
    return data_1

def parse_nested_json(text: str) -> Dict:
    a = text.strip()
    json_data = a.strip().replace('```json', '').strip()
    json_data = json_data.strip().replace('```', '').strip()
    data = json.loads(json_data)
    return data


def build_prompt(inputs:list, outputs:dict, template:str, include_parser: bool = True) -> PromptTemplate:
    response_schema = [ResponseSchema(name=k, description=outputs[k])\
            for k in outputs]
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions = output_parser.get_format_instructions()
    if include_parser:
        prompt = PromptTemplate(template=template, input_variables=inputs,\
                          output_parser=output_parser,\
                          partial_variables={"format_instructions": format_instructions})
    else:
        prompt = PromptTemplate(template=template, input_variables=inputs,\
                          partial_variables={"format_instructions": format_instructions})
    return prompt


class CheckupMemory():

    def __init__(self, input_key):
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key=input_key)

    def load_conversation(self, ingredient_str: str, day_number: int):
        self.memory.chat_memory.messages.pop()
        if day_number < 2:
            self.memory.chat_memory.add_ai_message(f"Welcome to day {day_number}! I am a nutrition doctor who can help achiving health goals effectively.")
        else:
            self.memory.chat_memory.add_ai_message(f"Welcome to day {day_number}!")
        self.memory.chat_memory.add_ai_message(f"Here is my advise. {ingredient_str}.")

    def get(self):
        return self.memory

class QAMemory():

    def __init__(self, input_key: str = 'problem'):
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key=input_key)
    
    def get(self):
        return self.memory

    def load_qa(self, questions: List[str], answers: List[str]):
        self.memory.chat_memory.add_ai_message("Hello! I am a nutrition doctor who can help achiving health goals effectively.")
        self.memory.chat_memory.add_user_message("Hello doctor, I need your help.")
        for q, a in zip(questions, answers):
            self.memory.chat_memory.add_ai_message(q)
            self.memory.chat_memory.add_user_message(a)
        return True
