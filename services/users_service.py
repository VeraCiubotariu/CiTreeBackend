from persistence.users_repository import UserRepository
from utils.custom_exceptions import ServiceException


class UserService:
    def __init__(self, users_repository: UserRepository):
        self.users_repository: UserRepository = users_repository

    def signup(self, user):
        return self.users_repository.add(user)

    def login(self, username, password):
        user = self.users_repository.get(username)

        if user.password != password:
            raise ServiceException('Wrong password!')

        return user