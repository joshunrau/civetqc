class Subject:

    def __init__(self, prefix, id) -> None:
        self.prefix = str(prefix)
        self.id = str(id)
    
    def __str__(self) -> str:
        return f"{self.prefix} {self.id}"