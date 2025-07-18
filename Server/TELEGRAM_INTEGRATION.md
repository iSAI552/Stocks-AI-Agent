# Telegram Integration for Stock AI Agent

This document explains how to set up and use the Telegram notification system for the Stock AI Agent.

## 🚀 Features

- **Automated Notifications**: Receive stock predictions directly in your Telegram chat
- **Formatted Messages**: Clean, emoji-rich formatting for easy reading
- **Message Chunking**: Handles large prediction data by splitting into multiple messages
- **Error Handling**: Robust error handling with admin notifications
- **User Personalization**: Personalized messages with user names

## 📋 Prerequisites

1. **Telegram Bot**: Create a bot using [@BotFather](https://t.me/botfather)
2. **Chat ID**: Get your Telegram chat ID
3. **Environment Variables**: Set up the required configuration

## 🛠️ Setup Instructions

### 1. Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow the instructions to create your bot
4. Save the **Bot Token** you receive

### 2. Get Your Chat ID

**Method 1: Using @userinfobot**
1. Search for [@userinfobot](https://t.me/userinfobot) in Telegram
2. Start the bot (`/start`)
3. It will show your chat ID

**Method 2: Using your bot**
1. Send a message to your newly created bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for the `chat.id` in the response

### 3. Environment Configuration

Create a `.env` file in the `Server/` directory:

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# User Configuration
USER_TELEGRAM_ID=123456789
USER_NAME=Your Name

# Admin Configuration (optional)
ADMIN_TELEGRAM_ID=987654321

# Other API keys...
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langchain_key
```

## 📝 Usage Examples

### Basic Usage

```python
from app.rag.graph import call_graph

# Run analysis with Telegram notifications
response = call_graph(
    user_telegram_id="123456789",
    user_name="John Doe"
)
```

### Advanced Usage

```python
import os
from dotenv import load_dotenv
from app.rag.graph import call_graph

# Load environment variables
load_dotenv()

# Get user info from environment
user_id = os.getenv("USER_TELEGRAM_ID")
user_name = os.getenv("USER_NAME", "User")

# Run the complete pipeline
response = call_graph(
    user_telegram_id=user_id,
    user_name=user_name
)
```

### Testing the Integration

Run the test script:

```bash
cd Server/
python test_telegram_integration.py
```

## 📱 Message Format

The Telegram notifications include:

### Stock Predictions
```
📈 STOCK PREDICTIONS FOR [USER NAME]

🏢 RELIANCE
📊 Action: BUY
🎯 Target: ₹2,500
🛡️ Stop Loss: ₹2,200
✅ Confidence: 85%
⏰ Horizon: 3 months
⚠️ Risk: Medium
💡 Reason: Strong fundamentals...
```

### Market Outlook
```
🌍 MARKET OUTLOOK
The overall market shows positive momentum...
```

### Risk Assessment
```
⚠️ RISK ASSESSMENT
Current market conditions suggest...
```

## 🔧 API Reference

### send_predictions_to_telegram()

```python
from app.resources.utils.telegram import send_predictions_to_telegram

success = send_predictions_to_telegram(
    chat_id="123456789",
    predictions_data={
        "final_predictions": {...},
        "market_outlook": "...",
        "overall_risk_assessment": "...",
        "initial_stock_recommendations": [...],
        "news": [...]
    },
    user_name="John Doe"
)
```

### send_alert_to_telegram()

```python
from app.resources.utils.telegram import send_alert_to_telegram

send_alert_to_telegram(
    chat_id="123456789",
    title="Alert Title",
    message="Alert message",
    alert_type="INFO"  # INFO, WARNING, ERROR, SUCCESS
)
```

## 🛡️ Error Handling

The system includes comprehensive error handling:

1. **Missing Chat ID**: Falls back to admin notifications
2. **API Failures**: Retry logic with exponential backoff
3. **Large Messages**: Automatic chunking for Telegram limits
4. **Network Issues**: Graceful degradation with console logging

## 🔒 Security Considerations

1. **Bot Token**: Keep your bot token secure and never commit it to version control
2. **Chat IDs**: Validate chat IDs to prevent unauthorized access
3. **Admin Notifications**: Use admin chat ID for system alerts
4. **Rate Limiting**: Respect Telegram's API rate limits

## 📚 Dependencies

Required packages:
```
requests>=2.25.0
python-dotenv>=0.19.0
```

Install with:
```bash
pip install requests python-dotenv
```

## 🐛 Troubleshooting

### Common Issues

1. **Bot not responding**
   - Check if bot token is correct
   - Ensure bot is started (`/start` command)

2. **Messages not received**
   - Verify chat ID is correct
   - Check if bot is blocked by user

3. **API errors**
   - Check internet connection
   - Verify bot token permissions

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For issues and questions:
1. Check the logs for error messages
2. Verify environment variable configuration
3. Test with the provided test script
4. Review Telegram Bot API documentation

## 🚀 Future Enhancements

Planned features:
- Interactive buttons for trade execution
- Real-time price alerts
- Portfolio tracking notifications
- Custom notification preferences
- Multi-language support
