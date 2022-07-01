import torch.nn as nn
import torch.nn.functional as F

    
class Regression(nn.Module):
    
    def __init__(self):
        super(Regression, self).__init__()
        
        self.l1 = nn.Linear(800*800*3, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, 52)
        self.l4 = nn.Linear(52, 2)
        
    def forward(self, inputs):
        x = inputs.view(800*800*3)
        x = self.l1(x)
        x = F.tanh (x)
        x = self.l2(x)
        x = F.tanh (x)
        x = self.l3(x)
        x = F.tanh (x)
        x = self.l4(x)
        return x


