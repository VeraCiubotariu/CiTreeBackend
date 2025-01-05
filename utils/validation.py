from domain.user import User
from utils.custom_exceptions import ValidationException


def validate_user(user: User):
    if user.username.replace(' ', '') == '' or ',' in user.username:
        raise ValidationException('Invalid username!')

    if len(user.password) < 8:
        raise ValidationException('Password must be at least 8 characters!')

    if ',' in user.password:
        raise ValidationException('Password cannot contain the character ","!')

    if user.firstName.replace(' ', '') == '' or ',' in user.firstName:
        raise ValidationException('Invalid first name!')

    if user.lastName.replace(' ', '') == '' or ',' in user.lastName:
        raise ValidationException('Invalid last name!')
