import json
from flask_login import UserMixin
from app import bcrypt, login_manager

class User(UserMixin):
    def __init__(self, username, password, role, hash_pasword=False):
        self.username = username
        if hash_pasword:
            self.password = bcrypt.generate_password_hash(password).decode('utf-8')
        else:
            self.password = password
        self.role = role

    def get_id(self):
        # Return a unique identifier for the user
        return self.username
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

@login_manager.user_loader
def load_user(username):
    users = load_users()
    return next((u for u in users if u.username == username), None)

class Role:
    def __init__(self, name):
        self.name = name

def load_users():
    with open('app/config/users.json') as f:
        users_data = json.load(f)
    users = []
    for user_data in users_data:
        users.append(User(user_data['username'], user_data['password'], user_data['role']))
    return users

def load_roles():
    with open('app/config/roles.json') as f:
        roles_data = json.load(f)
    roles = []
    for role_data in roles_data:
        roles.append(Role(role_data['name']))
    return roles

def save_users(users):
    users_data = [{'username': u.username, 'password': u.password, 'role': u.role} for u in users]
    with open('app/config/users.json', 'w') as f:
        json.dump(users_data, f)