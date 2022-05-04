class Score:

    def __init__(self, name, score, solution) -> None:
        self.name = name
        self.score = score
        self.solution = solution

    def __repr__(self) -> str:
        name = self.name + " "*(16-len(self.name))
        return f"{name}: {self.score:.5f}"