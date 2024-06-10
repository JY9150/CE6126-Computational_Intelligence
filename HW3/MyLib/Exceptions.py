class DimensionError(Exception):
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class OutputFileMismatch(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)