from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal, List, Dict, Optional
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import json

load_dotenv()

OPENAI_MINI_MODEL = ""
OPENAI_MAIN_MODEL = ""
GEMINI_MINI_MODEL = "gemini-2.5-flash-preview-04-17"
GEMINI_MAIN_MODEL = "gemini-2.5-pro"

ModelType = Literal["openai_mini", "openai_main", "gemini_mini", "gemini_main"]

clientOpenAI = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
clientGemini = wrap_openai(OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
))

# Dynamically construct the path to bucketStocks.json relative to this file
bucket_stocks_path = os.path.join(os.path.dirname(__file__), '../resources/bucketStocks.json')
with open(os.path.abspath(bucket_stocks_path), "r") as f:
    bucketStocks = json.load(f)

def llm_call(prompt: str, model: ModelType, responseFormat: BaseModel):
    """
    Calls the LLM with the given prompt and model, returns the parsed responseFormat object.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if model == "openai_mini" or model == "openai_main":
        client = clientOpenAI
        model = OPENAI_MINI_MODEL if model == "openai_mini" else OPENAI_MAIN_MODEL
    else:
        client = clientGemini
        model = GEMINI_MINI_MODEL if model == "gemini_mini" else GEMINI_MAIN_MODEL
    response = client.beta.chat.completions.parse(
        model=model,
        response_format=responseFormat,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.choices or not response.choices[0].message:
        raise ValueError("No valid response from LLM.")
    print("LLM Response:", response.choices[0].message.content.strip())
    # Parse the response as JSON and return the parsed object
    try:
        parsed = responseFormat.model_validate_json(response.choices[0].message.content.strip())
        return parsed
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return response.choices[0].message.content.strip()

class newsResponse(BaseModel):
    news: List[str]

class State(TypedDict):
    """
    Represents a state in the graph.
    """
    user_msg: str
    ai_msg: str
    news: Optional[List[str]]
    stock_bucket: Optional[List[Dict[str, Dict[str, str]]]]
    stock_recommendations: Optional[List[Dict[str, Dict[str, str]]]]

#TODO: remeber to add maybe a starting node that sets a system promt and make the ai a good stock ai assistant etc.

def get_overall_market_relevant_news(state: State) -> State:
    """
    Geth top 5 overall stock market relevant news.
    """
    prompt_overall_market_news = (
        "You are an Indian and US stock market authentic news aggregator and expert."
        "Provide me the top 5 news of the past 48 hours that can have impact on the Indian stock market."
        "The news should be relevant and correct also refrain from making individul stock based news."
        "Return these 5 news in an array of strings, each string being a news item."

        "Example Response:"
        """news: [
  "SEBI has barred U.S. trading firm Jane Street from Indian securities markets, accusing it of index-derivatives manipulation and ordering a ~$570 million escrow, highlighting increasing regulatory scrutiny in the derivatives space.",
  "Robust U.S. non-farm payrolls and a lower unemployment rate strengthened the dollar and U.S. Treasury yields, leading to a weaker rupee (near ₹85.5/USD) and pressuring FPIs to reduce Indian equity and bond holdings.",
  "The Reserve Bank of India conducted ₹1 trillion in 7-day VRRR operations to absorb surplus liquidity after cutting the policy rate by 50 bps, aiming to keep short-term rates within the policy corridor and support transmission.",
  "India's stainless-steel producers have requested an anti-dumping investigation into low-priced imports from China, Vietnam, and Malaysia—potentially triggering import duties and affecting metal and manufacturing-sector sentiment.",
  "U.S. sanctions were imposed on a shipping network engaged in transporting Iranian oil to China; although not directly India-specific, rising Middle East tensions could influence global oil prices and input costs."
]"""
    )
    news_obj = llm_call(prompt_overall_market_news, "gemini_mini", newsResponse)
    state["news"] = news_obj.news if hasattr(news_obj, "news") else []
    return state

def get_bucket_stocks_specific_news(state: State) -> State:
    """
    Get top 10 news specific to the bucket of stocks.
    """
    state["stock_bucket"] = bucketStocks
    prompt_stock_specific_news = (
        "You are an Indian and US stock market authentic news aggregator and expert."
        "Provide me the top 10 news of the past 48 hours that can have impact on any of the stocks provided in the bucket."
        "The news should be relevant and correct and should be related to individual or some of stocks in the bucket only."
        "Always Return these 10 news in an array of strings, each string being a news item. Each string enclosed in double quotes."
        f"bucket: {bucketStocks}"

        # "Example Response:"
        # """news: [
        #         "some reliance news",
        #         "maybe some more news related to reliance.",
        #         "Infosys related.",
        #         "some news impacting tcs, airtel, banks",
        #         "news affecting all the banks, hdfc, icici, sbi",
        #         "news affecting small caps like pvr, five star, etc.",
        #         "govt policy news affecting the bucket stocks market",
        #         "some quater earning news of Kotak bank which will lead to its fall",
        #         "PolyCab India ceo resigns, stock may fall",
        # ]"""
    )
    news_obj = llm_call(prompt_stock_specific_news, "gemini_main", newsResponse)
    if hasattr(news_obj, "news"):
        if state["news"] is None:
            state["news"] = []
        state["news"].extend(news_obj.news)
    return state

graph_builder = StateGraph(State)

graph_builder.add_node(
    "get_overall_market_relevant_news",
    get_overall_market_relevant_news,
)

graph_builder.add_node(
    "get_bucket_stocks_specific_news",
    get_bucket_stocks_specific_news,
)

graph_builder.add_edge(
    START,
    "get_overall_market_relevant_news",
)

graph_builder.add_edge(
    "get_overall_market_relevant_news",
    "get_bucket_stocks_specific_news",
)

graph_builder.add_edge(
    "get_bucket_stocks_specific_news",
    END,
)

graph = graph_builder.compile()

# Use the graph
def call_graph() -> str:
    """
    Calls the graph with the user message and returns the AI response.
    """
    state = {
        "user_msg": "",
        "ai_msg": "",
    }
    
    result = graph.invoke(state)
    return result.get("news", "No AI response generated.")

if __name__ == "__main__":
    print("Welcome to the Stock Market AI Assistant!")
    try:
        response = call_graph()
        print(f"news Response: {response}")
    except Exception as e:
        print(f"Error found: {e}")