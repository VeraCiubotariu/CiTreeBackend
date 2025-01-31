from flask_cors import CORS
from flask import Flask

from persistence.tree_repository import TreeRepository
from persistence.users_repository import UserRepository
from services.users_service import UserService

user_repo: UserRepository = UserRepository()
user_service: UserService = UserService(user_repo)
trees_repo: TreeRepository = TreeRepository()
