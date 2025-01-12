class Tree:
    def __init__(self, user_id, id, name, datePlanted, treeType, location):
        self.user_id = user_id
        self.id = id
        self.name = name
        self.datePlanted = datePlanted
        self.treeType = treeType
        self.location = location

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'datePlanted': self.datePlanted,
            'treeType': self.treeType,
            'location': {
                'lat': self.location.lat,
                'lng': self.location.lng,
            }
        }
