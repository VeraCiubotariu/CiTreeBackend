import json

from flask import Flask, request, jsonify
from flask_cors import CORS

from domain.user import User
from rest.flask_app import user_service


app = Flask(__name__)
CORS(app)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    try:
        user = user_service.login(data['username'], data['password'])
        response = jsonify(isError=False, message='Login successful', statusCode=201, data=json.dumps(user.to_dict())), 201
    except Exception as e:
        response = jsonify(isError=True, message=str(e), statusCode=401), 401

    return response

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    try:
        user = user_service.signup(User(data['username'], data['firstName'], data['lastName'], data['password']))
        return jsonify(isError=False, message='Signup successful', statusCode=201, data=json.dumps(user.to_dict())), 201
    except Exception as e:
        return jsonify(isError=True, message=str(e), statusCode=401), 401

app.run(debug=True)