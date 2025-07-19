# Multi-User Stock Predictions System

This document explains the enhanced functionality that allows the Stock Market AI Assistant to send predictions to multiple users automatically.

## Overview

The system now reads user details from `users.list.json` and sends stock predictions to all active users via Telegram. This eliminates the need to manually specify individual users and enables scalable notification distribution.

## Key Features

### 🎯 Automated Multi-User Notifications
- Reads user list from `app/resources/users.list.json`
- Sends personalized predictions to all active users
- Tracks notification timestamps
- Provides comprehensive admin summaries

### 👥 User Management
- Add/remove users programmatically
- Activate/deactivate users without deletion
- Track user preferences and settings
- Monitor notification history

### 📊 Enhanced Monitoring
- Admin notifications for success/failure summaries
- Individual user notification tracking
- Error handling and reporting
- System status monitoring

## File Structure

```
Server/
├── app/
│   ├── resources/
│   │   └── users.list.json          # User database
│   └── rag/
│       └── graph.py                 # Main application (modified)
├── manage_users.py                  # User management CLI tool
├── test_multi_user.py              # Testing script
└── README_MULTI_USER.md            # This documentation
```

## User Data Structure

Each user in `users.list.json` has the following structure:

```json
{
  "user_id": "unique_user_identifier",
  "user_name": "Display Name",
  "telegram_id": "telegram_chat_id",
  "email": "user@example.com",
  "access_to_holdings": true,
  "risk_tolerance": "medium",
  "investment_horizon": "long",
  "is_active": true,
  "last_notification": "2024-07-19T10:30:00"
}
```

### Field Descriptions:
- `user_id`: Unique identifier for the user
- `user_name`: Display name for personalized messages
- `telegram_id`: Telegram chat ID for notifications
- `email`: User's email address (optional)
- `access_to_holdings`: Whether user has portfolio analysis access
- `risk_tolerance`: User's risk preference (low/medium/high)
- `investment_horizon`: Investment timeframe (short/medium/long)
- `is_active`: Whether user should receive notifications
- `last_notification`: Timestamp of last successful notification

## Usage

### Running Stock Analysis
```bash
# Run the main application (sends to all active users)
cd Server
python -m app.rag.graph
```

### User Management
```bash
# List all active users
python manage_users.py active

# List all users (including inactive)
python manage_users.py list --include-inactive

# Add a new user interactively
python manage_users.py add

# Toggle user status (activate/deactivate)
python manage_users.py toggle
```

### Testing
```bash
# Run test suite (safe - no actual notifications)
python test_multi_user.py
```

## Code Changes Made

### 1. Enhanced `graph.py`
- Added `load_users_list()` function
- Modified `send_notification()` for multi-user support
- Added `update_user_notification_timestamp()`
- Added utility functions for user management
- Updated main execution flow

### 2. New User Management Scripts
- `manage_users.py`: CLI tool for user administration
- `test_multi_user.py`: Testing and demonstration script

### 3. User Database
- `users.list.json`: Centralized user configuration

## Admin Features

### Notification Summary
After each run, admins receive a summary via Telegram:
```
📊 Stock Predictions Notification Summary:
✅ Successful: 2
❌ Failed: 0
📋 Total Users: 2
```

### Error Handling
- Individual user errors don't stop the process
- Failed notifications are logged and reported
- Admin receives alerts for system issues
- Graceful handling of missing or invalid data

## Security Considerations

1. **Telegram ID Protection**: Store Telegram IDs securely
2. **User Data Privacy**: Limit access to user configuration files
3. **Error Information**: Avoid exposing sensitive data in error messages
4. **Admin Notifications**: Ensure admin Telegram ID is properly secured

## Migration from Single User

If you were previously using the single-user version:

1. **Environment Variables**: The system no longer reads `USER_TELEGRAM_ID` and `USER_NAME` from environment variables
2. **User Data**: Add your existing user to `users.list.json`
3. **Admin ID**: Ensure `ADMIN_TELEGRAM_ID` is set in your environment

### Migration Example:
```bash
# Old way (environment variables)
export USER_TELEGRAM_ID="123456789"
export USER_NAME="John Doe"

# New way (add to users.list.json)
python manage_users.py add
# Then enter your details when prompted
```

## Troubleshooting

### Common Issues:

1. **No Users Found**
   - Check if `users.list.json` exists
   - Verify users have `is_active: true`
   - Ensure proper JSON formatting

2. **Telegram Notifications Failing**
   - Verify Telegram bot token is valid
   - Check user Telegram IDs are correct
   - Ensure users have started the bot

3. **Permission Errors**
   - Check file permissions for `users.list.json`
   - Ensure write access for timestamp updates

### Debug Commands:
```bash
# Check user list
python -c "from app.rag.graph import load_users_list; print(load_users_list())"

# Test user addition
python test_multi_user.py

# Validate JSON format
python -c "import json; print(json.load(open('app/resources/users.list.json')))"
```

## Future Enhancements

Possible improvements for future versions:
- Database integration (PostgreSQL, MongoDB)
- Web-based user management interface
- Advanced user segmentation and targeting
- A/B testing for different prediction formats
- User preference learning and personalization
- Integration with external user management systems

## Support

For issues or questions about the multi-user functionality:
1. Check the logs for specific error messages
2. Verify user data format and structure
3. Test with a single user first
4. Review admin notification messages
5. Consult the troubleshooting section above
