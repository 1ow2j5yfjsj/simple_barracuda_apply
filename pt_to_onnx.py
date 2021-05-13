import torch
import torch.nn as nn
import torch.nn.functional as F

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



net = Test1()
net.load_state_dict(torch.load("sl_model_test.pt"))

# Input to the model
x = torch.randn(1, 1)

# Export the model
torch.onnx.export(net,                       # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "example.onnx",            # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,           # the ONNX version to export the model to
                  input_names = ['obs_0'], 
                  output_names = ["discrete_actions"],
                  dynamic_axes={'obs_0' : {0 : 'batch_size'},    # variable lenght axes
                                'discrete_actions' : {0 : 'batch_size'}}
                                )       # the model's output names