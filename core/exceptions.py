class ResponseParsingError(Exception):
    """Raised when the model response cannot be parsed."""

    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


class FunctionCallError(Exception):
    """Raised when a function call fails."""
    pass
