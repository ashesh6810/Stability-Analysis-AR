import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
#from prettytable import PrettyTable
#from count_trainable_params import count_parameters
import hdf5storage
import pickle
import h5py

path_outputs = '/media/volume/sdb/RK4_analysis/KS_stuff/new_outputs/'
f = h5py.File('/media/volume/sdb/RK4_analysis/KS_stuff/new_outputs/predicted_FNO_KS_directstep_lead1.mat','r')


data = np.asarray(f.get('prediction'))

print(np.shape(data))






time_step = 1e-3 
lead = int((time_step)/(1e-3))
trainN=80000
input_size = 512
output_size = 512
hidden_layer_size = 1000
eq_points = 11


input_test_torch = torch.from_numpy(np.transpose(data)).float().cuda()
print('shape of input_data in torch',input_test_torch.shape)
x_torch = torch.zeros([eq_points,input_size])


count=0
for k in (np.array([ int(1),  int(100), int(500), int(1000), int(3000), int(5000), int(7000), int(9000), int(10000), int(15000), int(19000)])):
 x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
 count=count+1


'''
count=0
for k in (np.array([ int(1)])):
 x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
 count=count+1
'''

def RK4step(input_batch):
 output_1 = mynet(input_batch.cuda())
 output_2 = mynet(input_batch.cuda()+0.5*output_1)
 output_3 = mynet(input_batch.cuda()+0.5*output_2)
 output_4 = mynet(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(mynet(input_batch.cuda())+2*(mynet(input_batch.cuda()+0.5*mynet(input_batch.cuda())))+2*(mynet(input_batch.cuda()+mynet(input_batch.cuda()+0.5*mynet(input_batch.cuda()))))+(mynet(input_batch.cuda())+mynet(input_batch.cuda()+mynet(input_batch.cuda()+0.5*mynet(input_batch.cuda())))))/6


 

def Eulerstep(input_batch):
 return input_batch.cuda() + time_step*mynet(input_batch.cuda())



def directstep(input_batch):
  return  mynet(input_batch.cuda())
  
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, time_future, time_history):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: a driving function observed at T timesteps + 1 locations (u(1, x), ..., u(T, x),  x).
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.modes = modes
        self.width = width
        self.time_future = time_future
        self.time_history = time_history
        self.fc0 = nn.Linear(self.time_history+1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

    def forward(self, u):
        grid = self.get_grid(u.shape, u.device)
        x = torch.cat((u, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu of no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 1

mynet = FNO1d(modes, width, time_future, time_history).cuda()
mynet.load_state_dict(torch.load('BNN_FNO_directstep_lead1.pt'))
mynet.eval()
mynet.cuda()

ygrad = torch.zeros([eq_points,input_size,input_size])

for k in range(0,eq_points):

    ygrad [k,:,:] = torch.autograd.functional.jacobian(directstep,torch.reshape(x_torch[k,:],(1,input_size,1))).squeeze()

ygrad = ygrad.detach().cpu().numpy()

print(ygrad.shape)



matfiledata = {}
matfiledata[u'A_matrix_FNO_direct'] = ygrad
hdf5storage.write(matfiledata, '.', path_outputs+'Amatrix_KS_FNO_directstep_11_ICs_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved Predictions')

