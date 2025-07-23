from Server.app.db.models import fundamental
from pydantic import BaseModel, Field   
from Server.app.db.db.connection import collection


def create_fundamental_schema():
    """
    Create a Pydantic schema for the fundamental model.
    """
    return fundamental.model_json_schema()
def save_fundamental_data(data: dict):
    """
    Save fundamental data to the MongoDB collection.
    
    :param data: A dictionary containing fundamental data.
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")
    
    # Insert the data into the collection
    result = collection.insert_one(data)
    return result.inserted_id
def get_fundamental_data(symbol: str):
    """
    Retrieve fundamental data for a given stock symbol from the MongoDB collection.
    :param symbol: The stock symbol to retrieve data for.
    :return: A dictionary containing the fundamental data for the specified symbol.
    """
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")
    
    # Find the document with the specified symbol
    data = collection.find_one({"symbol": symbol})
    if data is None:
        raise ValueError(f"No data found for symbol: {symbol}")
    return data
def update_fundamental_data(symbol: str, data: dict):
    """
    Update fundamental data for a given stock symbol in the MongoDB collection.
    :param symbol: The stock symbol to update data for.
    :param
    data: A dictionary containing the updated fundamental data.
    :return: The updated document.
    """
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")
    
    # Update the document with the specified symbol
    result = collection.update_one({"symbol": symbol}, {"$set": data})
    if result.matched_count == 0:
        raise ValueError(f"No data found for symbol: {symbol}")
    
    return collection.find_one({"symbol": symbol})
def delete_fundamental_data(symbol: str):
    """
    Delete fundamental data for a given stock symbol from the MongoDB collection.
    :param symbol: The stock symbol to delete data for.
    :return: The deleted document.
    """
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")
    
    # Delete the document with the specified symbol
    result = collection.find_one_and_delete({"symbol": symbol})
    if result is None:
        raise ValueError(f"No data found for symbol: {symbol}")
    
    return result
def list_fundamental_data():
    """
    List all fundamental data from the MongoDB collection.
    :return: A list of dictionaries containing fundamental data.
    """
    data = list(collection.find())
    return data
def count_fundamental_data():
    """
    Count the number of documents in the fundamental data collection.
    :return: The count of documents in the collection.
    """
    count = collection.count_documents({})
    return count