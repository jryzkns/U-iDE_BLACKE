import torch.nn.functional as F
import torch.nn as nn

class BLACkp(nn.Module):
    def __init__(self):
        super(BLACkp, self).__init__()

        self.fc1 = nn.Linear(75,200)
        self.fc2 = nn.Linear(200,128)
        self.fc3 = nn.Linear(128,4)

    def forward(self,x):
        
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out