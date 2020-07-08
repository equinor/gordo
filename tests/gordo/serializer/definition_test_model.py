class DefinitionTestModel:
    @classmethod
    def from_definition(cls, definition: dict):
        return cls(int(definition.get("depth", 10)))

    def __init__(self, depth):
        self.depth = depth

    def into_definition(self):
        return {"depth": self.depth}
