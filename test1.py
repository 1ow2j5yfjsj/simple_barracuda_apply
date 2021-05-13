import torch
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F

import numpy as np
from torch.nn.modules import loss

class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()
        self.last = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x, softmax_dim=-1):
        x = self.last(x)
        return F.softmax(x, dim=softmax_dim)




torch.manual_seed(42)
np.random.seed(42)

model = Test1()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()

print(optimizer.state_dict())

loss_function = nn.CrossEntropyLoss()

epochs = 10000

for i in range(epochs):
    print(f"{i} epoch start")

    x = np.random.randint(10, size=100)
    y = np.copy(x)
    x = x.reshape(-1, 1)

    x = torch.tensor(x).float()
    y = torch.tensor(y).long()


    pred = model(x)
   
    loss = loss_function(pred, y)
    
    optimizer.zero_grad()
    loss.backward()

    #print(model.last[0].weight.grad)
    
    optimizer.step()

torch.save(model.state_dict(), f"sl_model_test.pt")