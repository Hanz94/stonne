import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.SimulatedConv2d(5,5,5,'$PATH_TO_STONNE/simulation_files/maeri_128mses_128_bw.cfg', 'dogsandcats_tile.txt', sparsity_ratio=0.0, stats_path='.', groups=1) 
    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
print(net)
input_test = torch.randn(5,50,50).view(-1,5,50,50)
result  = net(input_test)

