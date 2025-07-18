"""
Telegram Utility Module for Stock AI Agent

This module provides utilities for sending messages to users via Telegram Bot API.
Supports large message handling with automatic chunking, formatting, and error handling.

Requirements:
- Install: pip install requests python-telegram-bot
- Set environment variables: TELEGRAM_BOT_TOKEN
- Get bot token from @BotFather on Telegram
"""

import os
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Production-ready Telegram notification service for Stock AI Agent.
    
    Features:
    - Automatic message chunking for large messages
    - HTML and Markdown formatting support
    - Rate limiting and retry logic
    - Error handling and logging
    - Message templates for stock predictions
    """
    
    def __init__(self, bot_token: Optional[str] = None):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (optional, will use env var if not provided)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("Telegram bot token not found. Set TELEGRAM_BOT_TOKEN environment variable.")
        
        # Log successful initialization without exposing the token
        token_preview = f"{self.bot_token[:10]}...{self.bot_token[-4:]}" if len(self.bot_token) > 14 else "[TOKEN_HIDDEN]"
        logger.info(f"TelegramNotifier initialized with token: {token_preview}")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.max_message_length = 4096  # Telegram's message limit
        self.rate_limit_delay = 1  # Delay between messages to avoid rate limiting
        
    def _sanitize_error_message(self, message: str) -> str:
        """
        Remove sensitive information from error messages.
        
        Args:
            message: Original error message
            
        Returns:
            Sanitized error message
        """
        if self.bot_token and self.bot_token in message:
            return message.replace(self.bot_token, "[BOT_TOKEN_HIDDEN]")
        return message
        
    def _make_request(self, method: str, data: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        """
        Make HTTP request to Telegram API with retry logic.
        
        Args:
            method: API method name
            data: Request payload
            retries: Number of retry attempts
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{method}"
        
        for attempt in range(retries):
            try:
                response = requests.post(url, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                if result.get("ok"):
                    return result
                else:
                    # Hide sensitive bot token in error messages
                    safe_result = {k: v for k, v in result.items() if k != 'ok'}
                    logger.error(f"Telegram API error for method {method}: {safe_result}")
                    if attempt == retries - 1:
                        raise Exception(f"Telegram API error: {result.get('description', 'Unknown error')}")
                        
            except requests.exceptions.RequestException as e:
                # Create a safe error message without exposing the bot token
                safe_error_msg = self._sanitize_error_message(str(e))
                logger.warning(f"Request failed to Telegram API method {method} (attempt {attempt + 1}/{retries}): {safe_error_msg}")
                if attempt == retries - 1:
                    raise Exception(f"Failed to send Telegram message after {retries} attempts: {safe_error_msg}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return {}
    
    def _chunk_message(self, message: str, max_length: int = None) -> List[str]:
        """
        Split large message into chunks that fit Telegram's limits.
        
        Args:
            message: Message to split
            max_length: Maximum length per chunk
            
        Returns:
            List of message chunks
        """
        if max_length is None:
            max_length = self.max_message_length - 100  # Leave buffer for formatting
            
        if len(message) <= max_length:
            return [message]
        
        chunks = []
        current_chunk = ""
        
        # Split by lines to preserve formatting
        lines = message.split('\n')
        
        for line in lines:
            # If single line is too long, split it by words
            if len(line) > max_length:
                words = line.split(' ')
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= max_length:
                        current_line += (" " + word if current_line else word)
                    else:
                        if current_chunk + "\n" + current_line:
                            if len(current_chunk + "\n" + current_line) <= max_length:
                                current_chunk += ("\n" + current_line if current_chunk else current_line)
                            else:
                                chunks.append(current_chunk)
                                current_chunk = current_line
                        current_line = word
                
                if current_line:
                    if len(current_chunk + "\n" + current_line) <= max_length:
                        current_chunk += ("\n" + current_line if current_chunk else current_line)
                    else:
                        chunks.append(current_chunk)
                        current_chunk = current_line
            else:
                # Normal line processing
                if len(current_chunk + "\n" + line) <= max_length:
                    current_chunk += ("\n" + line if current_chunk else line)
                else:
                    chunks.append(current_chunk)
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def send_message(
        self, 
        chat_id: Union[int, str], 
        message: str, 
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
        chunk_if_needed: bool = True
    ) -> bool:
        """
        Send message to Telegram chat.
        
        Args:
            chat_id: Telegram chat ID or username
            message: Message text to send
            parse_mode: Message formatting (HTML, Markdown, or None)
            disable_web_page_preview: Disable link previews
            chunk_if_needed: Automatically split large messages
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            messages_to_send = [message]
            
            if chunk_if_needed and len(message) > self.max_message_length:
                messages_to_send = self._chunk_message(message)
                logger.info(f"Message split into {len(messages_to_send)} chunks")
            
            for i, msg_chunk in enumerate(messages_to_send):
                data = {
                    "chat_id": chat_id,
                    "text": msg_chunk,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": disable_web_page_preview
                }
                
                result = self._make_request("sendMessage", data)
                
                if i < len(messages_to_send) - 1:  # Add delay between chunks
                    time.sleep(self.rate_limit_delay)
                    
                logger.info(f"Message chunk {i + 1}/{len(messages_to_send)} sent successfully to {chat_id}")
            
            return True
            
        except Exception as e:
            safe_error_msg = self._sanitize_error_message(str(e))
            logger.error(f"Failed to send message to {chat_id}: {safe_error_msg}")
            return False
    
    def send_stock_predictions(
        self, 
        chat_id: Union[int, str], 
        predictions_data: Dict[str, Any],
        user_name: str = "User"
    ) -> bool:
        """
        Send formatted stock predictions to user.
        
        Args:
            chat_id: Telegram chat ID
            predictions_data: Dictionary containing prediction results from call_graph()
            user_name: User's name for personalization
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build formatted message
            message = f"""
🤖 <b>Stock AI Agent - Daily Predictions</b>
👤 Hello {user_name}!
📅 Generated: {current_time}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 <b>STOCK PREDICTIONS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            final_predictions = predictions_data.get("final_predictions", {})
            
            if final_predictions:
                for symbol, pred in final_predictions.items():
                    action_emoji = "🟢" if pred.get("action") == "BUY" else "🔴" if pred.get("action") == "SELL" else "🟡"
                    
                    message += f"""
{action_emoji} <b>{symbol}</b>
┌ Action: <b>{pred.get('action', 'N/A')}</b>
├ Price Target: {pred.get('price_target', 'N/A')}
├ Stop Loss: {pred.get('stop_loss', 'N/A')}
├ Confidence: {pred.get('confidence', 'N/A')}
├ Time Horizon: {pred.get('time_horizon', 'N/A')}
├ Risk Level: {pred.get('risk_level', 'N/A')}
├ Reason: {pred.get('reason', 'N/A')[:100]}{'...' if len(pred.get('reason', '')) > 100 else ''}
└ Key Factors: {', '.join(pred.get('key_factors', [])[:3])}

"""
            else:
                message += "\n❌ No predictions available at this time.\n"
            
            # Add market outlook
            message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌍 <b>MARKET OUTLOOK</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{predictions_data.get('market_outlook', 'No market outlook available')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ <b>RISK ASSESSMENT</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{predictions_data.get('overall_risk_assessment', 'No risk assessment available')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ <b>Disclaimer:</b> This is AI-generated analysis for educational purposes only. Always consult with a financial advisor before making investment decisions.

🤖 Powered by Stock AI Agent
"""
            
            return self.send_message(chat_id, message)
            
        except Exception as e:
            safe_error_msg = self._sanitize_error_message(str(e))
            logger.error(f"Failed to send stock predictions: {safe_error_msg}")
            return False
    
    def send_alert(
        self, 
        chat_id: Union[int, str], 
        title: str, 
        message: str, 
        alert_type: str = "INFO"
    ) -> bool:
        """
        Send formatted alert message.
        
        Args:
            chat_id: Telegram chat ID
            title: Alert title
            message: Alert message
            alert_type: Type of alert (INFO, WARNING, ERROR, SUCCESS)
            
        Returns:
            True if sent successfully, False otherwise
        """
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        
        emoji = emoji_map.get(alert_type, "ℹ️")
        formatted_message = f"""
{emoji} <b>{title}</b>

{message}

🕒 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        return self.send_message(chat_id, formatted_message)
    
    def test_connection(self, chat_id: Union[int, str]) -> bool:
        """
        Test Telegram connection by sending a test message.
        
        Args:
            chat_id: Telegram chat ID to test
            
        Returns:
            True if connection successful, False otherwise
        """
        test_message = f"""
🧪 <b>Test Message</b>

✅ Telegram connection is working!
🕒 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

🤖 Stock AI Agent is ready to send you predictions.
"""
        
        return self.send_message(chat_id, test_message)


# Global instance for easy import
telegram_notifier = TelegramNotifier()


# Convenience functions for easy usage
def send_predictions_to_telegram(chat_id: Union[int, str], predictions_data: Dict[str, Any], user_name: str = "User") -> bool:
    """
    Convenience function to send stock predictions to Telegram.
    
    Args:
        chat_id: Telegram chat ID
        predictions_data: Prediction results from call_graph()
        user_name: User's name
        
    Returns:
        True if sent successfully, False otherwise
    """
    return telegram_notifier.send_stock_predictions(chat_id, predictions_data, user_name)


def send_alert_to_telegram(chat_id: Union[int, str], title: str, message: str, alert_type: str = "INFO") -> bool:
    """
    Convenience function to send alert to Telegram.
    
    Args:
        chat_id: Telegram chat ID
        title: Alert title
        message: Alert message
        alert_type: Alert type (INFO, WARNING, ERROR, SUCCESS)
        
    Returns:
        True if sent successfully, False otherwise
    """
    return telegram_notifier.send_alert(chat_id, title, message, alert_type)


def send_message_to_telegram(chat_id: Union[int, str], message: str, parse_mode: str = "HTML") -> bool:
    """
    Convenience function to send a simple message to Telegram.
    
    Args:
        chat_id: Telegram chat ID
        message: Message text
        parse_mode: Message formatting
        
    Returns:
        True if sent successfully, False otherwise
    """
    return telegram_notifier.send_message(chat_id, message, parse_mode)


# # Example usage
# if __name__ == "__main__":
#     # Example usage - replace with actual chat ID
#     TEST_CHAT_ID = "7874696049"
    
#     # Test connection
#     print("Testing Telegram connection...")
#     if telegram_notifier.test_connection(TEST_CHAT_ID):
#         print("✅ Connection successful!")
#     else:
#         print("❌ Connection failed!")
    
#     # Example: Send stock predictions
#     sample_predictions = {
#         "final_predictions": {
#             "RELIANCE": {
#                 "action": "BUY",
#                 "price_target": "₹3200",
#                 "stop_loss": "₹2700", 
#                 "confidence": "85%",
#                 "time_horizon": "medium",
#                 "risk_level": "medium",
#                 "reason": "Strong fundamentals and positive market sentiment support upward movement.",
#                 "key_factors": ["Earnings growth", "Market expansion", "Sector rotation"]
#             }
#         },
#         "market_outlook": "Markets showing bullish momentum with sector rotation favoring energy stocks.",
#         "overall_risk_assessment": "Medium risk environment with selective opportunities in large-cap stocks."
#     }
    
#     send_predictions_to_telegram(TEST_CHAT_ID, sample_predictions, "John")
