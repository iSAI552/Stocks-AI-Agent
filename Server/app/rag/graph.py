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
from datetime import datetime
from ..utils.telegram import send_alert_to_telegram, send_predictions_to_telegram

load_dotenv()

OPENAI_MINI_MODEL = ""
OPENAI_MAIN_MODEL = ""
GEMINI_MINI_MODEL = "gemini-2.5-flash"
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

def load_users_list():
    """
    Load the list of users from users.list.json file.
    Returns a list of active users with their details.
    """
    users_list_path = os.path.join(os.path.dirname(__file__), '../resources/users.list.json')
    try:
        with open(os.path.abspath(users_list_path), "r") as f:
            users_data = json.load(f)
        
        # Filter only active users
        active_users = [user for user in users_data if user.get("is_active", True)]
        print(f"Loaded {len(active_users)} active users from users list")
        return active_users
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading users list: {e}")
        return []

def update_user_notification_timestamp(user_telegram_id: str):
    """
    Update the last_notification timestamp for a specific user.
    """
    users_list_path = os.path.join(os.path.dirname(__file__), '../resources/users.list.json')
    try:
        with open(os.path.abspath(users_list_path), "r") as f:
            users_data = json.load(f)
        
        # Update the timestamp for the specific user
        current_time = datetime.now().isoformat()
        for user in users_data:
            if user.get("telegram_id") == user_telegram_id:
                user["last_notification"] = current_time
                break
        
        # Write back to file
        with open(os.path.abspath(users_list_path), "w") as f:
            json.dump(users_data, f, indent=2)
    
    except Exception as e:
        print(f"Error updating notification timestamp: {e}")

def add_new_user(user_data: Dict) -> bool:
    """
    Add a new user to the users.list.json file.
    
    Args:
        user_data: Dictionary containing user information
                  Required: user_id, user_name, telegram_id
                  Optional: email, access_to_holdings, risk_tolerance, etc.
    
    Returns:
        bool: True if user was added successfully, False otherwise
    """
    users_list_path = os.path.join(os.path.dirname(__file__), '../resources/users.list.json')
    try:
        # Load existing users
        try:
            with open(os.path.abspath(users_list_path), "r") as f:
                users_data = json.load(f)
        except FileNotFoundError:
            users_data = []
        
        # Check if user already exists
        for user in users_data:
            if user.get("user_id") == user_data.get("user_id") or user.get("telegram_id") == user_data.get("telegram_id"):
                print(f"User with ID {user_data.get('user_id')} or Telegram ID {user_data.get('telegram_id')} already exists")
                return False
        
        # Set default values for new user
        new_user = {
            "user_id": user_data.get("user_id"),
            "user_name": user_data.get("user_name"),
            "telegram_id": user_data.get("telegram_id"),
            "email": user_data.get("email", ""),
            "access_to_holdings": user_data.get("access_to_holdings", True),
            "risk_tolerance": user_data.get("risk_tolerance", "medium"),
            "investment_horizon": user_data.get("investment_horizon", "long"),
            "is_active": user_data.get("is_active", True),
            "last_notification": None
        }
        
        # Add new user
        users_data.append(new_user)
        
        # Write back to file
        with open(os.path.abspath(users_list_path), "w") as f:
            json.dump(users_data, f, indent=2)
        
        print(f"Successfully added new user: {new_user['user_name']} ({new_user['user_id']})")
        return True
        
    except Exception as e:
        print(f"Error adding new user: {e}")
        return False

def deactivate_user(user_identifier: str, by_telegram_id: bool = True) -> bool:
    """
    Deactivate a user in the users.list.json file.
    
    Args:
        user_identifier: Either user_id or telegram_id
        by_telegram_id: If True, search by telegram_id, otherwise by user_id
    
    Returns:
        bool: True if user was deactivated successfully, False otherwise
    """
    users_list_path = os.path.join(os.path.dirname(__file__), '../resources/users.list.json')
    try:
        with open(os.path.abspath(users_list_path), "r") as f:
            users_data = json.load(f)
        
        # Find and deactivate user
        user_found = False
        search_field = "telegram_id" if by_telegram_id else "user_id"
        
        for user in users_data:
            if user.get(search_field) == user_identifier:
                user["is_active"] = False
                user_found = True
                print(f"Deactivated user: {user.get('user_name')} ({user.get('user_id')})")
                break
        
        if not user_found:
            print(f"User with {search_field} {user_identifier} not found")
            return False
        
        # Write back to file
        with open(os.path.abspath(users_list_path), "w") as f:
            json.dump(users_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error deactivating user: {e}")
        return False

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

class FinalPrediction(BaseModel):
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    price_target: str
    stop_loss: str
    confidence: str
    time_horizon: str  # "short", "medium", "long"
    risk_level: str  # "low", "medium", "high"
    reason: str
    key_factors: List[str]

class finalPredictionResponse(BaseModel):
    final_predictions: List[FinalPrediction]
    market_outlook: str
    overall_risk_assessment: str

class State(TypedDict):
    """
    Represents a state in the graph.
    """
    user_msg: str
    ai_msg: str
    access_to_holdings: bool
    news: Optional[List[str]]
    stock_bucket: Optional[List[Dict[str, Dict[str, str]]]]
    stock_recommendations: Optional[List[Dict[str, Dict[str, str]]]]
    stock_fundamentals: Optional[Dict[str, str]]
    stock_technicals: Optional[Dict[str, str]]
    stock_sentiments: Optional[Dict[str, str]]
    stock_mutual_funds: Optional[Dict[str, str]]
    existing_stock_holdings: Optional[Dict[str, Dict[str, str]]]
    final_predictions: Optional[Dict[str, Dict[str, str]]]
    market_outlook: Optional[str]
    overall_risk_assessment: Optional[str]
    user_telegram_id: Optional[str]
    user_name: Optional[str]

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

def stock_sentiment_analysis(state: State) -> State:
    """
    Perform sentiment analysis on the stocks recommended.
    """

    #  prompt_to_get_summarised_from_json_sentiments = (""""Perform a sentiment analysis for the stock [STOCK_NAME] (ticker: 
    # [TICKER]) listed on [MARKET, e.g., India (NSE/BSE) or US (NYSE/NASDAQ)] and provide the results in JSON format.
    # Include a detailed breakdown of positive, negative, and neutral factors influencing the sentiment, along with sources,
    # impact levels, and references (e.g., web articles, X posts). Ensure the JSON includes a 'sentiment_summary' section with a
    # concise overview of the sentiment, including positive, negative, and neutral drivers, and the overall outlook.
    # Then, extract the 'sentiment_summary' section and return it as a single string, summarizing the key drivers and outlook in
    # a clear and concise manner."""")


    # Initialize stock_fundamentals if it doesn't exist
    if state.get("stock_sentiments") is None:
        state["stock_sentiments"] = {}
    
    # Get the base path for fundamentals directory
    sentiments_base_path = os.path.join(os.path.dirname(__file__), '../resources/sentiments')
    
    # Process each stock recommendation
    for stock_rec in state["stock_recommendations"].stock_recommendations:
        symbol = stock_rec.symbol
        print(f"Processing sentiments for stock: {symbol}")
        symbol = "RELIANCE" #TODO: this is kept for testing purposes, remove later.
        
        # Construct the path to the fundamentals file
        sentiments_file_path = os.path.join(sentiments_base_path, f"{symbol}.sentiments.json")
        
        try:
            # Check if file exists and read it
            if os.path.exists(sentiments_file_path):
                with open(sentiments_file_path, 'r') as f:
                    sentiments_data = json.load(f)
                    
                # Extract summary and add to state
                summary = sentiments_data.get("summary", "No summary available")
                state["stock_sentiments"][symbol] = summary
            else:
                print(f"Warning: Sentiments file not found for {symbol}")
                state["stock_sentiments"][symbol] = "Sentiments data not available"
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading sentiments for {symbol}: {e}")
            state["stock_sentiments"][symbol] = "Error loading sentiments data"
    print("Final stock sentiments:", state["stock_sentiments"])
    return state

def stock_mutual_funds_weightage_analysis(state: State) -> State:
    """
    Perform mutual funds weightage analysis on the stocks recommended.
    """
    #  prompt_to_get_summarised_from_json_mf = ("""Create a concise summary (maximum 5 lines) of the provided
    # JSON data for a specified stock, capturing all relevant details for an LLM to understand the stock's current 
    # position in major Indian mutual funds (e.g., Parag Parikh Flexi Cap, Motilal Oswal, Kotak Flexicap, SBI Equity 
    # Hybrid, ICICI Prudential Bluechip). Include the stock's mutual fund holding percentage, specific fund holdings 
    # with weightage changes (e.g., increase/decrease with percentages or qualitative notes), current share price, 
    # 52-week high/low, P/E ratio, market cap, average target price, analyst ratings (buy/hold/sell), and key growth 
    # drivers. Highlight any underperformance or recovery potential and mention data sources with their dates. 
    # Ensure the summary is dense with information, structured for clarity, and suitable for LLM processing.""")


    # Initialize stock_fundamentals if it doesn't exist
    if state.get("stock_mutual_funds") is None:
        state["stock_mutual_funds"] = {}
    
    # Get the base path for fundamentals directory
    mutual_funds_base_path = os.path.join(os.path.dirname(__file__), '../resources/mutualFundsData')
    
    # Process each stock recommendation
    for stock_rec in state["stock_recommendations"].stock_recommendations:
        symbol = stock_rec.symbol
        print(f"Processing mfs for stock: {symbol}")
        symbol = "RELIANCE" #TODO: this is kept for testing purposes, remove later.
        
        # Construct the path to the fundamentals file
        mutual_funds_file_path = os.path.join(mutual_funds_base_path, f"{symbol}.mutualFund.json")
        
        try:
            # Check if file exists and read it
            if os.path.exists(mutual_funds_file_path):
                with open(mutual_funds_file_path, 'r') as f:
                    mf_data = json.load(f)
                    
                # Extract summary and add to state
                summary = mf_data.get("summary", "No summary available")
                state["stock_mutual_funds"][symbol] = summary
            else:
                print(f"Warning: mf file not found for {symbol}")
                state["stock_mutual_funds"][symbol] = "MF data not available"
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading MF for {symbol}: {e}")
            state["stock_mutual_funds"][symbol] = "Error loading MF data"
    print("Final stock MF:", state["stock_mutual_funds"])
    return state
    
def check_holdings_access(state: State) -> Literal["analyse_mystock_holdings", "combined_prediction"]:
    """
    Conditional function to determine next node based on access_to_holdings flag.
    """
    if state.get("access_to_holdings", False):
        return "analyse_mystock_holdings"
    else:
        return "combined_prediction"

def analyse_mystock_holdings(state: State) -> State:
    """
    Perform analysis on the user's stock holdings.
    Optional Node.
    """
    # make the database call here or get the data from user input 

    user_stock_holdings = [
        {
            "symbol": "RELIANCE",
            "quantity": "50",
            "averageBuyPrice": "2800",
            "purchaseDates": ["2024-01-15"],
            "targetPrice": "3200",
            "stopLossPrice": "2700",
            "portfolioAllocation": "25",
            "investmentHorizon": "long",
            "dividendPreference": "True",
            "sector": "Energy",
            "riskTolerance": "medium"
        },
        {
            "symbol": "TCS",
            "quantity": "20",
            "averageBuyPrice": "3500",
            "purchaseDates": ["2024-03-10"],
            "targetPrice": "4000",
            "stopLossPrice": "3400",
            "portfolioAllocation": "15",
            "investmentHorizon": "medium",
            "dividendPreference": "False",
            "sector": "IT",
            "riskTolerance": "low"
        }
    ]

    # Initialize existing_stock_holdings if it doesn't exist
    if state.get("existing_stock_holdings") is None:
        state["existing_stock_holdings"] = {}

    # Process each stock recommendation
    for stock_rec in state["stock_recommendations"].stock_recommendations:
        symbol = stock_rec.symbol
        print(f"Processing holdings for stock: {symbol}")
        symbol = "RELIANCE"
        # Check if the stock is in the user's holdings
        for holding in user_stock_holdings:
            if holding["symbol"] == symbol:
                # Add the holding details to the state
                state["existing_stock_holdings"][symbol] = {
                    "quantity": holding["quantity"],
                    "averageBuyPrice": holding["averageBuyPrice"],
                    "purchaseDates": holding["purchaseDates"],
                    "targetPrice": holding["targetPrice"],
                    "stopLossPrice": holding["stopLossPrice"],
                    "portfolioAllocation": holding["portfolioAllocation"],
                    "investmentHorizon": holding["investmentHorizon"],
                    "dividendPreference": holding["dividendPreference"],
                    "sector": holding["sector"],
                    "riskTolerance": holding["riskTolerance"]
                }
                break
    
    print("Final existing stock holdings:", state["existing_stock_holdings"])           

    return state

def combined_prediction(state: State) -> State:
    """
    Combine all analyses and generate final stock predictions.
    """
    # Initialize final_predictions if it doesn't exist
    if state.get("final_predictions") is None:
        state["final_predictions"] = {}
    
    # Prepare comprehensive prompt with all available analysis data
    prompt_combined_analysis = f"""
    You are a world-class stock market analyst and investment advisor with expertise in Indian and US markets. 
    You have been provided with comprehensive analysis data for multiple stocks. Your task is to generate final actionable investment predictions for today ({datetime.now().strftime('%Y-%m-%d')}).

    ANALYSIS DATA PROVIDED:
    ======================

    NEWS ANALYSIS:
    {state.get('news', 'No news data available')}

    INITIAL STOCK RECOMMENDATIONS:
    {state.get('stock_recommendations', 'No recommendations available')}

    FUNDAMENTAL ANALYSIS:
    {state.get('stock_fundamentals', 'No fundamental data available')}

    TECHNICAL ANALYSIS:
    {state.get('stock_technicals', 'No technical data available')}

    SENTIMENT ANALYSIS:
    {state.get('stock_sentiments', 'No sentiment data available')}

    MUTUAL FUNDS ANALYSIS:
    {state.get('stock_mutual_funds', 'No mutual funds data available')}

    USER STOCK HOLDINGS (if available):
    {state.get('existing_stock_holdings', 'No user holdings data available')}

    INSTRUCTIONS:
    ============

    Based on ALL the analysis data above, provide comprehensive final investment predictions. For each stock:

    1. Analyze and synthesize ALL available data (news, fundamentals, technicals, sentiment, mutual funds, user holdings)
    2. Provide a clear ACTION: BUY, SELL, or HOLD
    3. Set realistic PRICE TARGETS and STOP LOSS levels
    4. Assess CONFIDENCE level (0-100%)
    5. Determine appropriate TIME HORIZON (short/medium/long term)
    6. Evaluate RISK LEVEL (low/medium/high)
    7. Provide detailed REASONING incorporating all analysis factors
    8. List KEY FACTORS that influenced your decision

    ADDITIONAL REQUIREMENTS:
    - Consider current market conditions and broader economic factors
    - If user holdings data is available, provide personalized advice (whether to add, reduce, or maintain positions)
    - Factor in risk management and portfolio diversification
    - Be realistic about price targets and timeframes
    - Highlight any conflicting signals between different analysis types

    MARKET OUTLOOK:
    Provide an overall market outlook for the next 1-3 months based on the news and analysis.

    RISK ASSESSMENT:
    Provide an overall risk assessment for the current market environment.

    Return your analysis in the specified JSON format with final_predictions, market_outlook, and overall_risk_assessment.
    """

    try:
        # Make LLM call to get final predictions
        final_response = llm_call(
            prompt_combined_analysis,
            "gemini_main",  # Using main model for critical final analysis
            finalPredictionResponse
        )
        
        # Store the final predictions in state
        if hasattr(final_response, 'final_predictions'):
            # Convert to dictionary format for easier access
            final_preds_dict = {}
            for pred in final_response.final_predictions:
                final_preds_dict[pred.symbol] = {
                    "action": pred.action,
                    "price_target": pred.price_target,
                    "stop_loss": pred.stop_loss,
                    "confidence": pred.confidence,
                    "time_horizon": pred.time_horizon,
                    "risk_level": pred.risk_level,
                    "reason": pred.reason,
                    "key_factors": pred.key_factors
                }
            
            state["final_predictions"] = final_preds_dict
            
            # Store market outlook and risk assessment
            state["market_outlook"] = getattr(final_response, 'market_outlook', 'No market outlook provided')
            state["overall_risk_assessment"] = getattr(final_response, 'overall_risk_assessment', 'No risk assessment provided')
        
        print("Final predictions generated successfully")
        print(f"Market Outlook: {state.get('market_outlook', 'N/A')}")
        print(f"Risk Assessment: {state.get('overall_risk_assessment', 'N/A')}")
        
    except Exception as e:
        print(f"Error generating final predictions: {e}")
        state["final_predictions"] = {"error": "Failed to generate final predictions"}
        state["market_outlook"] = "Unable to assess market outlook due to error"
        state["overall_risk_assessment"] = "Unable to assess risk due to error"
    
    return state

def update_final_values_in_db(state: State) -> State:
    """
    Update the final predictions and analysis in the database.
    This is a placeholder function for actual database update logic.
    """
    # Here you would implement the logic to update the database with final predictions
    # For now, we will just print the final predictions
    print("Updating final predictions in database...")
    
    # Example of how you might structure the update
    for symbol, prediction in state["final_predictions"].items():
        print(f"Updating {symbol}: {prediction}")
    
    return state

def send_notification(state: State) -> State:
    """
    Send final predictions to all users in the users list via Telegram.
    """
    try:
        # Load users list
        users_list = load_users_list()
        
        if not users_list:
            print("❌ No active users found in users list")
            admin_id = os.getenv("ADMIN_TELEGRAM_ID")
            if admin_id:
                send_alert_to_telegram(
                    chat_id=admin_id,
                    title="No Users Found",
                    message="No active users found in users.list.json - predictions generated but not sent",
                    alert_type="WARNING"
                )
            return state
        
        # Prepare prediction data for Telegram
        predictions_data = {
            "final_predictions": state.get("final_predictions", {}),
            "market_outlook": state.get("market_outlook", "No market outlook available"),
            "overall_risk_assessment": state.get("overall_risk_assessment", "No risk assessment available"),
            "initial_stock_recommendations": state.get("stock_recommendations", []),
            "news": state.get("news", [])
        }
        
        # Send notifications to all users
        successful_notifications = 0
        failed_notifications = 0
        
        for user in users_list:
            user_telegram_id = user.get("telegram_id")
            user_name = user.get("user_name", "User")
            user_id = user.get("user_id", "unknown")
            
            if not user_telegram_id:
                print(f"⚠️ No Telegram ID for user {user_name} ({user_id}) - skipping")
                failed_notifications += 1
                continue
            
            try:
                # Send predictions to user via Telegram
                success = send_predictions_to_telegram(
                    chat_id=user_telegram_id,
                    predictions_data=predictions_data,
                    user_name=user_name
                )
                
                if success:
                    print(f"✅ Successfully sent predictions to {user_name} ({user_telegram_id})")
                    successful_notifications += 1
                    # Update the user's notification timestamp
                    update_user_notification_timestamp(user_telegram_id)
                else:
                    print(f"❌ Failed to send predictions to {user_name} ({user_telegram_id})")
                    failed_notifications += 1
                    
            except Exception as user_error:
                print(f"❌ Error sending to {user_name} ({user_telegram_id}): {user_error}")
                failed_notifications += 1
        
        # Send summary to admin
        admin_id = os.getenv("ADMIN_TELEGRAM_ID")
        if admin_id:
            summary_message = (
                f"📊 Stock Predictions Notification Summary:\n"
                f"✅ Successful: {successful_notifications}\n"
                f"❌ Failed: {failed_notifications}\n"
                f"📋 Total Users: {len(users_list)}"
            )
            
            send_alert_to_telegram(
                chat_id=admin_id,
                title="Notification Summary",
                message=summary_message,
                alert_type="SUCCESS" if failed_notifications == 0 else "WARNING"
            )
        
        print(f"📊 Notification Summary: {successful_notifications} successful, {failed_notifications} failed out of {len(users_list)} users")
    
    except Exception as e:
        print(f"❌ Critical error in send_notification: {e}")
        
        # Send error to admin if available
        admin_id = os.getenv("ADMIN_TELEGRAM_ID")
        if admin_id:
            send_alert_to_telegram(
                chat_id=admin_id,
                title="Critical Notification Error",
                message=f"Exception in send_notification function: {str(e)}",
                alert_type="ERROR"
            )
    
    return state

def self_improve(state: State) -> State:
    """
    Self-improvement function to enhance the AI's performance.
    This is a placeholder for future self-improvement logic.
    """
    # Implement self-improvement logic here
    print("Self-improvement function called - no implementation yet")

    # This is an future work optional Node
    # This basically will check at the end of the market session whether something actually happened as predicted or not.
    # Then make a memory database of it how and what happened actually.
    # Also can include auto promt engineering to improve the prompts used in the analysis.
    # Later point this can be used to change the stock bucket list of 30 we have.
    
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

graph_builder.add_node(
    "stock_sentiment_analysis",
    stock_sentiment_analysis,
)

graph_builder.add_node(
    "stock_mutual_funds_weightage_analysis",
    stock_mutual_funds_weightage_analysis,
)

graph_builder.add_node(
    "analyse_mystock_holdings",
    analyse_mystock_holdings,
)

graph_builder.add_node(
    "combined_prediction",
    combined_prediction,
)

graph_builder.add_node(
    "update_final_values_in_db",
    update_final_values_in_db,
)

graph_builder.add_node(
    "send_notification",
    send_notification,
)

graph_builder.add_node(
    "self_improve",
    self_improve,
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
    "stock_sentiment_analysis",
)

graph_builder.add_edge(
    "stock_sentiment_analysis",
    "stock_mutual_funds_weightage_analysis",
)

graph_builder.add_conditional_edges(
    "stock_mutual_funds_weightage_analysis",
    check_holdings_access,
    {
        "analyse_mystock_holdings": "analyse_mystock_holdings",
        "combined_prediction": "combined_prediction"
    }
)

graph_builder.add_edge(
    "analyse_mystock_holdings",
    "combined_prediction",
)

graph_builder.add_edge(
    "combined_prediction",
    "update_final_values_in_db",
)

graph_builder.add_edge(
    "update_final_values_in_db",
    "send_notification",
)

graph_builder.add_edge(
    "send_notification",
    "self_improve",
)

graph_builder.add_edge(
    "self_improve",
    END,
)

graph = graph_builder.compile()

# Use the graph
def call_graph() -> Dict:
    """
    Calls the graph and returns the AI response.
    Users are loaded from users.list.json file.
    """
    state = {
        "user_msg": "",
        "ai_msg": "",
        "access_to_holdings": True,  # Assuming we have access to user's holdings
    }
    
    result: State = graph.invoke(state)
    
    # Return comprehensive results
    return {
        "final_predictions": result.get("final_predictions", {}),
        "market_outlook": result.get("market_outlook", "No market outlook available"),
        "overall_risk_assessment": result.get("overall_risk_assessment", "No risk assessment available"),
        "initial_stock_recommendations": result.get("stock_recommendations", []),
        "news": result.get("news", [])
    }

if __name__ == "__main__":
    print("Welcome to the Stock Market AI Assistant!")
    try:
        # Load users list to show how many users will receive notifications
        users_list = load_users_list()
        print(f"Found {len(users_list)} active users in the system")
        
        for user in users_list:
            print(f"  - {user.get('user_name', 'Unknown')} (ID: {user.get('user_id', 'N/A')})")
        
        print("\nRunning stock market analysis...")
        
        response = call_graph()
        
        print(f"\n=== FINAL STOCK PREDICTIONS ===")
        for symbol, pred in response["final_predictions"].items():
            print(f"\nStock: {symbol}")
            print(f"Action: {pred['action']}")
            print(f"Price Target: {pred['price_target']}")
            print(f"Stop Loss: {pred['stop_loss']}")
            print(f"Confidence: {pred['confidence']}")
            print(f"Time Horizon: {pred['time_horizon']}")
            print(f"Risk Level: {pred['risk_level']}")
            print(f"Reason: {pred['reason']}")
            print(f"Key Factors: {', '.join(pred['key_factors'])}")
        
        print(f"\n=== MARKET OUTLOOK ===")
        print(response["market_outlook"])
        
        print(f"\n=== OVERALL RISK ASSESSMENT ===")
        print(response["overall_risk_assessment"])
        
        print(f"\n✅ Analysis complete! Notifications sent to all active users via Telegram.")
        
    except Exception as e:
        print(f"Error found: {e}")
        # Send error to admin if available
        admin_id = os.getenv("ADMIN_TELEGRAM_ID")
        if admin_id:
            send_alert_to_telegram(
                chat_id=admin_id,
                title="System Error",
                message=f"Critical error in main execution: {str(e)}",
                alert_type="ERROR"
            )