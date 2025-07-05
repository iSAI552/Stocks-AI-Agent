from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal, List
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

OPENAI_MINI_MODEL = ""
OPENAI_MAIN_MODEL = ""
GEMINI_MINI_MODEL = ""
GEMINI_MAIN_MODEL = ""

ModelType = Literal["openai_mini", "openai_main", "gemini_mini", "gemini_main"]

clientOpenAI = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
clientGemini = wrap_openai(OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
))

def llm_call(prompt: str, model: ModelType) -> str:
    """
    Calls the LLM with the given prompt and model.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    
    if model == "openai_mini" or model == "openai_main":
        client = clientOpenAI
        model = OPENAI_MINI_MODEL if model == "openai_mini" else OPENAI_MAIN_MODEL
    else:
        client = clientGemini
        model = GEMINI_MINI_MODEL if model == "gemini_mini" else GEMINI_MAIN_MODEL

    # class DetectCallResponse(TypedDict):
    #     choices: List[dict]
    
    response = client.beta.chat.completions.create(
        model=model,
        # response_format=DetectCallResponse,
        messages=[{"role": "user", "content": prompt}],
    )
    
    if not response.choices or not response.choices[0].message:
        raise ValueError("No valid response from LLM.")
    
    return response.choices[0].message.content.strip()

class DetectCallResponse(BaseModel):
    pass

class State(TypedDict):
    """
    Represents a state in the graph.
    """
    model: ModelType
    prompt: str
    response: str
    error: str | None
