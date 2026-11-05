import numpy as np 


class SLP:
    def __init__(self,input_size, output_size):
        
 
        self.matriz_pesos= np.random.randn(input_size, output_size)
    
        self.bias = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

   

    def forward(self, X):
        self.final_input = np.dot(X, self.matriz_pesos) + self.bias
        
      
        
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output , taxa_de_aprendizado):
        ## erro quadrico 
        n = X.shape[0]

        erro_quadrático = (output - y)**2 
        derivada =  2 * (output - y) * (output) * (1 - output)
        self.matriz_pesos -= taxa_de_aprendizado * np.dot(X.T, derivada) 
        self.bias -= taxa_de_aprendizado * np.sum(derivada, axis=0, keepdims=True)  

    def train(self, X, y, epocas, taxa_de_aprendizado):
        for epoca in range(epocas):
            output = self.forward(X)
            self.backward(X, y, output, taxa_de_aprendizado)