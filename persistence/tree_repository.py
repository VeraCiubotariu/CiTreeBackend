import uuid

from domain.location import Location
from domain.tree import Tree
from utils.custom_exceptions import RepositoryException
from utils.validation import validate_tree


class TreeRepository:
    def __init__(self):
        self.trees = {}
        self._trees_file = '../resources/trees.txt'

        self._read_trees()

    def _read_trees(self):
        with open(self._trees_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line == '':
                    continue

                user_id, id, name, datePlanted, treeType, lat, lng = line.split(',')
                tree = Tree(user_id, id, name, datePlanted, treeType, Location(lat, lng))

                if self.trees.get(user_id) is None:
                    self.trees[user_id] = {}

                self.trees[user_id][id] = tree

    def _write_trees_to_file(self):
        with open(self._trees_file, 'w') as f:
            for user_id in self.trees:
                for id in self.trees[user_id]:
                    tree = self.trees[user_id][id]
                    f.write(f'{user_id},{id},{tree.name},{tree.datePlanted},{tree.treeType},{tree.location.lat},{tree.location.lng}\n')

    def get_all(self, user_id):
        if not self.trees.get(user_id) or self.trees[user_id] is None:
            return []

        return self.trees[user_id].values()

    def modify(self, new_tree):
        if new_tree.id not in self.trees[new_tree.user_id]:
            raise RepositoryException(f"Tree with id {new_tree.id} does not exist")

        validate_tree(new_tree)

        self.trees[new_tree.user_id][new_tree.id] = new_tree
        self._write_trees_to_file()

        return new_tree

    def add(self, new_tree):
        unique_id = str(uuid.uuid4())
        new_tree.id = unique_id

        validate_tree(new_tree)

        if self.trees.get(new_tree.user_id) is None:
            self.trees[new_tree.user_id] = {}

        self.trees[new_tree.user_id][unique_id] = new_tree
        self._write_trees_to_file()
        return new_tree
