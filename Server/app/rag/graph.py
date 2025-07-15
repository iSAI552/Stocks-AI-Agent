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

class StockRecommendation(BaseModel):
    symbol: str
    expected_change: str
    confidence: str
    reason: str

class stockRecommendationResponse(BaseModel):
    stock_recommendations: List[StockRecommendation]

    class Config:
        json_schema_extra = {
            "example": {
                "stock_recommendations": [
                    {
                        "symbol": "AAPL",
                        "expected_change": "2.5%",
                        "confidence": "85%",
                        "reason": "Positive earnings report."
                    },
                    {
                        "symbol": "GOOGL",
                        "expected_change": "-1.2%",
                        "confidence": "70%",
                        "reason": "Regulatory concerns."
                    }
                ]
            }
        }

class State(TypedDict):
    """
    Represents a state in the graph.
    """
    user_msg: str
    ai_msg: str
    news: Optional[List[str]]
    stock_bucket: Optional[List[Dict[str, Dict[str, str]]]]
    stock_recommendations: Optional[List[Dict[str, Dict[str, str]]]]
    stock_fundamentals: Optional[Dict[str, str]]
    stock_technicals: Optional[Dict[str, str]]

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
    # state["stock_bucket"] = bucketStocks
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

def map_news_to_stocks(state: State) -> State:
    """
    Maps the news to the stocks in the bucket.
    """
    if state["news"] is None:
        raise ValueError("News array cannot be empty.")
    
    prompt_map_news_to_stocks = (
        "You are an Stock Market and finance expert for Indian and US stock markets."
        "From the given newsArray and the bucketStocks, analyze the news very deliberately and give me the top 10 stocks that are most likely to be affected by the news."
        "The affected stocks should only be the ones present in the bucketStocks."
        "Return the top 10 stocks in an array of objects, each object containing the stock exchange symbol, the percentage fall or rise expected today on the stock because of this, a percentage showing how confident you are on this price change, and to the point reason for its impact in 2 lines max."
        f"newsArray: {state['news']}\n"
        f"bucketStocks: {bucketStocks}\n"
    )

    state["stock_recommendations"] = llm_call(
        prompt_map_news_to_stocks,
        "gemini_main",
        stockRecommendationResponse
    )
    return state

def check_for_similar_past_instances(state: State) -> State:
    """
    Check for similar past instances of the news and see how it impacted the current list of stocks.
    """
    prompt_check_past_instances = (
        "You are an Stock Market and finance expert and have deep understanding of the Indian and US Stock markets."
        "On the basis of the NEWS_ARRAY i have some predictins in STOCK_RECOMMENDATIONS, analyze the past instances of similar news and how they impacted the specific stock i have provided or extremly similar stocks."
        "Now if you get some similar news and based on this new data you are very confident that the stock price change will be different that what is given, then only update the STOCK_RECOMMENDATIONS with the new expected change, confidence and append your reason for the change in the reason field."
        "Finally only return the STOCK_RECOMMENDATIONS with the updated values or if no update is required then return the same STOCK_RECOMMENDATIONS."
        "Return the 10 stocks in an array of objects, each object containing the stock exchange symbol, the percentage fall or rise expected today on the stock, a percentage showing how confident you are on this price change now, and appended reason for its impact in 2 lines max."
        f"NEWS_ARRAY: {state['news']}\n"
        f"STOCK_RECOMMENDATIONS: {state['stock_recommendations']}\n"
    )

    state["stock_recommendations"] = llm_call(
        prompt_check_past_instances,
        "gemini_main",
        stockRecommendationResponse
    )
    return state

def stock_fundamental_analysis(state: State) -> State:
    """
    Perform fundamental analysis on the stocks recommended.
    """

    #  prompt_to_get_summarised_from_json_fundamentals = ("""You are a professional financial analyst assistant. You will receive a JSON containing detailed fundamental analysis data for a publicly traded company. Your task is to generate a high-quality, concise summary (maximum 4 lines) that captures the most critical insights from the data.
# Your summary must:
# Highlight overall financial strength using key metrics like revenue, profit, growth rates, and valuation multiples (P/E, market cap).
# Emphasize the performance of major business segments with numbers (e.g., retail, digital, O2C) and relevant user/customer metrics.
# Mention strategic initiatives or investments (e.g., mergers, energy, IPOs), especially those with long-term impact.
# Call out any important risks or uncertainties (e.g., leadership succession, execution delays, IPO timeline).
# Include analyst sentiment and expected upside/downside if available.
# Use precise numbers from the JSON (e.g., revenue of ₹9.98 lakh crore, ARPU ₹206.2, Jio users 488M, etc.) and synthesize the insights naturally as a human would in a financial executive briefing.
# Output only the final summary in clear, well-structured prose. Do not include bullet points, analysis steps, or raw data. Keep it within 4 lines, highly informative, and suitable for downstream use in stock prediction pipelines.""")


    # Initialize stock_fundamentals if it doesn't exist
    if state.get("stock_fundamentals") is None:
        state["stock_fundamentals"] = {}
    
    # Get the base path for fundamentals directory
    fundamentals_base_path = os.path.join(os.path.dirname(__file__), '../resources/fundamentals')
    
    # Process each stock recommendation
    for stock_rec in state["stock_recommendations"].stock_recommendations:
        symbol = stock_rec.symbol
        print(f"Processing fundamentals for stock: {symbol}")
        symbol = "RELIANCE" #TODO: this is kept for testing purposes, remove later.
        
        # Construct the path to the fundamentals file
        fundamentals_file_path = os.path.join(fundamentals_base_path, f"{symbol}.fundamentals.json")
        
        try:
            # Check if file exists and read it
            if os.path.exists(fundamentals_file_path):
                with open(fundamentals_file_path, 'r') as f:
                    fundamentals_data = json.load(f)
                    
                # Extract summary and add to state
                summary = fundamentals_data.get("summary", "No summary available")
                state["stock_fundamentals"][symbol] = summary
            else:
                print(f"Warning: Fundamentals file not found for {symbol}")
                state["stock_fundamentals"][symbol] = "Fundamentals data not available"
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading fundamentals for {symbol}: {e}")
            state["stock_fundamentals"][symbol] = "Error loading fundamentals data"
    print("Final stock fundamentals:", state["stock_fundamentals"])
    return state

def stock_technical_analysis(state: State) -> State:
    """
    Perform technical analysis on the stocks recommended.
    """

    # prompt_to_get_summarised_from_json_technicals = ("Given the following technical analysis in JSON format for a stock",
    # "generate a concise summary (maximum 5 lines) with clear, actionable insights for use in an AI-based trading workflow."
    # "The summary should include: current trend, key support and resistance levels, important technical indicators (like RSI, MACD)",
    # "any identified chart patterns (such as channels, breakouts, etc.), and risk levels. Write the summary in a professional",
    # "human-like tone—similar to what a market analyst or trader might use. Avoid technical jargon or filler; focus on clarity"
    # "and precision to support decision-making.")



    # Initialize stock_fundamentals if it doesn't exist
    if state.get("stock_technicals") is None:
        state["stock_technicals"] = {}
    
    # Get the base path for fundamentals directory
    technicals_base_path = os.path.join(os.path.dirname(__file__), '../resources/technicals')
    
    # Process each stock recommendation
    for stock_rec in state["stock_recommendations"].stock_recommendations:
        symbol = stock_rec.symbol
        print(f"Processing technical for stock: {symbol}")
        symbol = "RELIANCE" #TODO: this is kept for testing purposes, remove later.
        
        # Construct the path to the fundamentals file
        technicals_file_path = os.path.join(technicals_base_path, f"{symbol}.technicals.json")
        
        try:
            # Check if file exists and read it
            if os.path.exists(technicals_file_path):
                with open(technicals_file_path, 'r') as f:
                    technicals_data = json.load(f)
                    
                # Extract summary and add to state
                summary = technicals_data.get("summary", "No summary available")
                state["stock_technicals"][symbol] = summary
            else:
                print(f"Warning: Technicals file not found for {symbol}")
                state["stock_technicals"][symbol] = "Technicals data not available"
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading fundamentals for {symbol}: {e}")
            state["stock_technicals"][symbol] = "Error loading technicals data"
    print("Final stock technicals:", state["stock_technicals"])
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

graph_builder.add_node(
    "map_news_to_stocks",
    map_news_to_stocks,
)

graph_builder.add_node(
    "check_for_similar_past_instances",
    check_for_similar_past_instances,
)

graph_builder.add_node(
    "stock_fundamental_analysis",
    stock_fundamental_analysis,
)

graph_builder.add_node(
    "stock_technical_analysis",
    stock_technical_analysis,
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
    "map_news_to_stocks",
)

graph_builder.add_edge(
    "map_news_to_stocks",
    "check_for_similar_past_instances",
)

graph_builder.add_edge(
    "check_for_similar_past_instances",
    "stock_fundamental_analysis",
)

graph_builder.add_edge(
    "stock_fundamental_analysis",
    "stock_technical_analysis",
)

graph_builder.add_edge(
    "stock_technical_analysis",
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
    
    result: State = graph.invoke(state)
    return result.get("stock_recommendations", "No AI response generated.")

if __name__ == "__main__":
    print("Welcome to the Stock Market AI Assistant!")
    try:
        response = call_graph()
        print(f"news Response: {response}")
    except Exception as e:
        print(f"Error found: {e}")