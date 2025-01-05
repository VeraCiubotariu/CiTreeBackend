from domain.user import User
from utils.custom_exceptions import RepositoryException
from utils.validation import validate_user

class UserRepository:
    def __init__(self):
        self.users = {}
        self._users_file = '../resources/users.txt'

        self._read_users()

    def _read_users(self):
        with open(self._users_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line == '':
                    continue

                username, first_name, last_name, password = line.split(',')
                user = User(username, first_name, last_name, password)

                self.users[username] = user

    def add(self, user):
        validate_user(user)

        if user.username in self.users.keys():
            raise RepositoryException(f'User with username "{user.username}" already exists')

        with open(self._users_file, 'a') as f:
            f.write(f'{user.username},{user.firstName},{user.lastName},{user.password}\n')

        self.users[user.username] = user
        return self.users[user.username]

    def get(self, username):
        if username not in self.users.keys():
            raise RepositoryException(f'User with username "{username}" does not exist')

        return self.users[username]
