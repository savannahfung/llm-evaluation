import time

class LLM:
    '''
    LLM class for creating and generating messages using the specified model and the given messages.

    Attributes:
        model (str): The model name.

    Note:
        This is an abstract class. The create and generate methods must be implemented in the derived class.
    '''
    model = ""

    def __init__(self, model):
        self.model = model

    def create(self, messages):
        pass

    def generate(self, messages):
        pass