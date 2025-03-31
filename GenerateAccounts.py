"""
GenerateAccounts.py

This script is used to generate user accounts with encrypted passwords. 
It supports both interactive mode and file-based input/output for account creation. 
The script uses bcrypt for password hashing and provides options for managing 
existing output files.

Features:
- Generate random passwords using three random words.
- Encrypt passwords using bcrypt.
- Load user accounts from an input JSON file.
- Save user accounts to an output JSON file.
- Interactive mode for manual account creation.
- Command-line arguments for input/output files and overwrite control.

Usage:
1. Run in interactive mode (default):
   python GenerateAccounts.py

2. Specify input and output files:
   python GenerateAccounts.py -i input.json -o output.json

3. Force overwrite of the output file:
   python GenerateAccounts.py -i input.json -o output.json -f

Arguments:
- -i, --input: Input file containing accounts to create (optional).
- -o, --output: Output file to save the accounts (required unless in interactive mode).
- -f, --force: Force overwrite of the output file without prompt (optional).

Dependencies:
- argparse: For command-line argument parsing.
- os: For file and directory operations.
- wonderwords: For generating random words.
- bcrypt: For password hashing.
- json: For reading and writing JSON files.

Author: Daniel Prince
Date: 27/03/25
Licence: Apache V2.0
"""
import argparse
import os
from wonderwords import RandomWord
import bcrypt
import json


# Function to generate a password from three random words
def generate_password():
    r = RandomWord()
    words = r.random_words(3, word_max_length=6, word_min_length=6)
    words = [word.capitalize() for word in words]
    return ''.join(words)

# Function to encrypt the password using bcrypt
def encrypt_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    return hashed

# Function to load accounts from an input file
def load_json_list(input_file):
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            return json.load(f)
    else:
        return []

# Function to save accounts to an output file
def save_accounts(output_file, users, force):
    if os.path.exists(output_file) and not force:
        confirm = input(f"{output_file} already exists. Overwrite? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
    with open(output_file, 'w') as f:
        json.dump(users, f, indent=2)
    print(f"Users saved to {output_file}")

def save_clear_text_passwords(output_file, users, force):
    if os.path.exists(output_file) and not force:
        confirm = input(f"{output_file} already exists. Overwrite? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
    # Save clear-text usernames and passwords to the specified file
    with open(args.cleartext, 'w') as f:
        for username, password in users.items():
            f.write(f"Username: {username}, Password: {password}\n")
    print(f"Clear-text usernames and passwords saved to {args.cleartext}")


# Interactive mode to create accounts manually
def interactive_mode(roles):
    users = []
    while True:
        username = input("Enter username (or 'done' to finish): ").strip()
        if username.lower() == 'done':
            break
        elif username.lower() == '':
            print("Username cannot be empty.")
            continue
        elif any(user['username'] == username for user in users):
            print(f'Username: {username} already exists')
            continue
        while True:
            print("Available roles are:")
            for index, role in enumerate(roles, start=1):
                print(f"{index}. {role}")
            try:
                role_index = int(input("Enter the number corresponding to the role for the user: ").strip())
                if role_index < 1 or role_index > len(roles):
                    print(f"Invalid selection. Please choose a number between 1 and {len(roles)}.")
                    continue
                role = roles[role_index - 1]
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            break
        users.append({"username": username, "role": role})
    return users

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate accounts with encrypted passwords.")
    parser.add_argument('-i', '--input', type=str, help="Input file containing accounts to create.")
    parser.add_argument('-o', '--output', default=r".\app\config\users.json", type=str, help="Output file to save the accounts.")
    parser.add_argument('-r', '--rolesinput', type=str, default=r".\app\config\roles.json", help="Input file to load the roles to for the accounts from.")
    parser.add_argument('-f', '--force', action='store_true', help="Force overwrite of the output file without prompt.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-c', '--cleartext', type=str, help="File to save clear-text usernames and passwords.")
    group.add_argument('-n', '--nocleartext', action='store_true', help='Forces the programme to not store the passwords into a clear text file.')
    args = parser.parse_args()

    # Handle default for --cleartext if not provided and --nocleartext is not set
    if not args.nocleartext and not args.cleartext:
        args.cleartext = "clear_text_passwords.txt"

    if args.rolesinput:
        roles = load_json_list(args.rolesinput)
        if not roles:
            print(f"Input file for roles '{args.rolesinput}' not found or empty.")
            exit(1)
        roles = [entry['name'].lower() for entry in roles]
    
    # Load accounts from input file or enter interactive mode
    if args.input:
        print (f'Loading input file {args.input}')
        users = load_json_list(args.input)
        if not users:
            print(f"Input file '{args.input}' not found or empty.")
            exit(1)
    else:
        print("Entering interactive mode.")
        users = interactive_mode(roles)

        if not users:
            print("No users created exiting")
            exit(0)
    
    # Generate and encrypt passwords for each user
    clear_text_users = {}
    for user in users:
        password = generate_password()
        clear_text_users[user['username']] = password
        print(f"Generated password for {user['username']}: {password}")
        encrypted_password = encrypt_password(password)
        user['password'] = encrypted_password

    # Save accounts to the output file
    save_accounts(args.output, users, args.force)

    if args.cleartext:
        save_clear_text_passwords(args.cleartext, clear_text_users, args.force)
