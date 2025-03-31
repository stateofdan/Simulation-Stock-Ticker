from flask import render_template, url_for, flash, redirect, request, session
from app import app, bcrypt, global_rng
from app.models import User, load_users, load_roles
from flask_login import login_user, current_user, logout_user, login_required
import json

users = load_users()
roles = load_roles()

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = next((u for u in users if u.username == username), None)
        if user and user.check_password(password):
            login_user(user)
            print (f'login:{user.username}, role:{user.role}')
            if user.role == "user":
                return redirect(url_for('user_dashboard'))

            return redirect(url_for('home'))
        else:
            print (f'login unsuccessful:{username}')
            flash('Login unsuccessful. Check username and password.', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/admin")
@login_required
def admin():
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    return render_template('admin.html')

@app.route("/diagnostics")
@login_required
def diagnostics():
    # Display all session cookie information
    session_data = {key: session[key] for key in session.keys()}
    return render_template('diagnostics.html', session_data=session_data)

@app.route("/user_dashboard")
@login_required
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route("/")
@app.route("/home")
def home():
    print("Accessing home page")
    return render_template('home.html')



