from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
import numpy as np
from app.stock_data import StockData
from helpers.config_loader import Config
import os

main_config_path = "./app/config/main_config.json"

app = Flask(__name__)
# This sets the secret key for signing the session cookie
app.config['SECRET_KEY'] = 'your_secret_key'
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
global_rng = np.random.default_rng(42)
print (f'working dir: {os.getcwd()}')
main_config = Config(main_config_path)
main_config.print_config_keys()
users_file = main_config.get('users_file')
group_role = main_config.get('group_role')
print (users_file, group_role)
group_stock_data = StockData(StockData.load_users(users_file, group_role))
from app import routes