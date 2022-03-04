class Vector:
    def __init__(self, inputs: list):
        self.inputs = inputs
    
    def dot(self, weigths):
        dot_product = 0
        for index in range(len(self.inputs)):
            dot_product = dot_product + (self.inputs[index] * weigths.inputs[index])
        return dot_product