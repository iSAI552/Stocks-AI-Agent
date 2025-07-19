#!/usr/bin/env python3
"""
User Management Script for Stock Market AI Assistant

This script provides command-line utilities to manage users:
- List all users
- Add new users
- Activate/Deactivate users
- View user notification history
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add the Server directory to the path so we can import from app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.rag.graph import (
    load_users_list, 
    add_new_user, 
    deactivate_user
)

def list_all_users(include_inactive=False):
    """List all users in the system"""
    users_list_path = os.path.join(os.path.dirname(__file__), 'app/resources/users.list.json')
    try:
        with open(os.path.abspath(users_list_path), "r") as f:
            users_data = json.load(f)
        
        print(f"📋 Total users in system: {len(users_data)}")
        print("=" * 80)
        
        for i, user in enumerate(users_data, 1):
            status = "🟢 Active" if user.get("is_active", True) else "🔴 Inactive"
            if not include_inactive and not user.get("is_active", True):
                continue
                
            print(f"{i}. {user.get('user_name', 'Unknown')} ({status})")
            print(f"   User ID: {user.get('user_id', 'N/A')}")
            print(f"   Telegram ID: {user.get('telegram_id', 'N/A')}")
            print(f"   Email: {user.get('email', 'N/A')}")
            print(f"   Risk Tolerance: {user.get('risk_tolerance', 'N/A')}")
            print(f"   Investment Horizon: {user.get('investment_horizon', 'N/A')}")
            print(f"   Holdings Access: {'Yes' if user.get('access_to_holdings', False) else 'No'}")
            
            last_notification = user.get('last_notification')
            if last_notification:
                try:
                    dt = datetime.fromisoformat(last_notification)
                    print(f"   Last Notification: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    print(f"   Last Notification: {last_notification}")
            else:
                print(f"   Last Notification: Never")
            print()
        
    except FileNotFoundError:
        print("❌ Users list file not found!")
    except json.JSONDecodeError:
        print("❌ Error reading users list file!")

def add_user_interactive():
    """Add a new user interactively"""
    print("🆕 Adding New User")
    print("=" * 30)
    
    user_data = {}
    
    # Required fields
    user_data['user_id'] = input("Enter User ID (unique): ").strip()
    if not user_data['user_id']:
        print("❌ User ID is required!")
        return False
    
    user_data['user_name'] = input("Enter User Name: ").strip()
    if not user_data['user_name']:
        print("❌ User Name is required!")
        return False
    
    user_data['telegram_id'] = input("Enter Telegram ID: ").strip()
    if not user_data['telegram_id']:
        print("❌ Telegram ID is required!")
        return False
    
    # Optional fields
    user_data['email'] = input("Enter Email (optional): ").strip()
    
    risk_tolerance = input("Enter Risk Tolerance (low/medium/high) [medium]: ").strip().lower()
    user_data['risk_tolerance'] = risk_tolerance if risk_tolerance in ['low', 'medium', 'high'] else 'medium'
    
    investment_horizon = input("Enter Investment Horizon (short/medium/long) [long]: ").strip().lower()
    user_data['investment_horizon'] = investment_horizon if investment_horizon in ['short', 'medium', 'long'] else 'long'
    
    holdings_access = input("Has access to holdings? (y/n) [y]: ").strip().lower()
    user_data['access_to_holdings'] = holdings_access != 'n'
    
    # Confirm before adding
    print("\n📋 User Details:")
    for key, value in user_data.items():
        print(f"  {key}: {value}")
    
    confirm = input("\nConfirm adding this user? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ User addition cancelled.")
        return False
    
    # Add the user
    success = add_new_user(user_data)
    if success:
        print("✅ User added successfully!")
    else:
        print("❌ Failed to add user!")
    
    return success

def toggle_user_status():
    """Activate or deactivate a user"""
    print("🔄 Toggle User Status")
    print("=" * 25)
    
    identifier = input("Enter User ID or Telegram ID: ").strip()
    if not identifier:
        print("❌ Identifier is required!")
        return False
    
    by_telegram = input("Search by Telegram ID? (y/n) [y]: ").strip().lower() != 'n'
    
    # Try to deactivate first (assuming they want to deactivate an active user)
    success = deactivate_user(identifier, by_telegram_id=by_telegram)
    
    if success:
        print("✅ User status updated successfully!")
    else:
        print("❌ Failed to update user status!")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Stock Market AI Assistant - User Management')
    parser.add_argument('action', choices=['list', 'add', 'toggle', 'active'], 
                       help='Action to perform')
    parser.add_argument('--include-inactive', action='store_true', 
                       help='Include inactive users when listing')
    
    args = parser.parse_args()
    
    print("🤖 Stock Market AI Assistant - User Management")
    print("=" * 50)
    
    if args.action == 'list':
        list_all_users(include_inactive=args.include_inactive)
    elif args.action == 'add':
        add_user_interactive()
    elif args.action == 'toggle':
        toggle_user_status()
    elif args.action == 'active':
        # Show only active users (same as list but filtered)
        active_users = load_users_list()
        print(f"📋 Active users: {len(active_users)}")
        print("=" * 30)
        for i, user in enumerate(active_users, 1):
            print(f"{i}. {user.get('user_name')} (ID: {user.get('user_id')})")
            print(f"   Telegram: {user.get('telegram_id')}")
            print(f"   Last notification: {user.get('last_notification', 'Never')}")
            print()

if __name__ == "__main__":
    main()
