class User:
    def __init__(self, username, first_name, last_name, password):
        self.username = username
        self.firstName = first_name
        self.lastName = last_name
        self.password = password

    def to_dict(self):
        return {
            "username": self.username,
            "firstName": self.firstName,
            "lastName": self.lastName,
            "password": self.password,
        }