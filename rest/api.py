import json
import os

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from domain.location import Location
from domain.tree import Tree
from domain.user import User
from rest.flask_app import user_service, trees_repo

app = Flask(__name__)
CORS(app)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    try:
        user = user_service.login(data['username'], data['password'])
        response = jsonify(message='Login successful', user=user.__dict__), 201
    except Exception as e:
        response = jsonify(message=str(e)), 401

    return response

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    try:
        user = user_service.signup(User(data['username'], data['firstName'], data['lastName'], data['password']))
        return jsonify(message='Signup successful', user=user.__dict__), 201
    except Exception as e:
        return jsonify(message=str(e)), 401

@app.route('/api/user/<string:user_id>/trees', methods=['GET'])
def get_trees(user_id):
    try:
        trees = trees_repo.get_all(user_id)
        trees_dict = [tree.to_dict() for tree in trees]
        print(user_id)
        print(trees_dict)
        return jsonify(trees=trees_dict), 200
    except Exception as e:
        return jsonify(message=str(e)), 404

@app.route('/api/user/<string:user_id>/trees', methods=['POST'])
def save_tree(user_id):
    data = request.get_json()
    try:
        new_tree = Tree(user_id, '', data['name'], data['datePlanted'], data['treeType'], Location(data['location']['lat'], data['location']['lng']))
        saved_tree = trees_repo.add(new_tree)
        return jsonify(tree=saved_tree.to_dict()), 200
    except Exception as e:
        return jsonify(message=str(e)), 404

@app.route('/api/user/<string:user_id>/trees/<string:tree_id>', methods=['PUT'])
def update_tree(user_id,tree_id):
    data = request.get_json()
    try:
        new_tree = Tree(user_id, tree_id, data['name'], data['datePlanted'], data['treeType'],
                        Location(data['location']['lat'], data['location']['lng']))
        saved_tree = trees_repo.modify(new_tree)
        return jsonify(tree=saved_tree.to_dict()), 200
    except Exception as e:
        return jsonify(message=str(e)), 404

@app.route('/api/get-mocked-target', methods=['GET'])
def get_mocked_target():
    image_path = "../resources/heatmap.png"

    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return {"error": "Image not found"}, 404

app.run(debug=True)