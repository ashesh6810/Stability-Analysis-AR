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
f = h5py.File('/media/volume/sdb/RK4_analysis/KS_stuff/new_outputs/predicted_KS_Spectral_Loss_with_tendencyfft_PECstep_lambda_reg_5_lead1.mat','r')
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
for k in (np.array([ int(0),  int(100), int(500), int(1000), int(3000), int(5000), int(7000), int(7000), int(10000), int(15000), int(19000)])):
 x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
 count=count+1


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


def PECstep(input_batch):
 output_1 = mynet(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_1))




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il  = ((nn.Linear(input_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.il.weight)

        self.hidden1  = ((nn.Linear(hidden_layer_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.hidden1.weight)

        self.hidden2  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden2.weight)

        self.hidden3  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden3.weight)

        self.hidden4  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden4.weight)

        self.hidden5  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden5.weight)

        self.ol  = nn.Linear(hidden_layer_size,output_size)
        torch.nn.init.xavier_uniform_(self.ol.weight)

        self.tanh = nn.Tanh()


    def forward(self,x):

        x1 = self.tanh(self.il(x))
        x2 = self.tanh(self.hidden1(x1))
        x3 = self.tanh(self.hidden2(x2))
        x4 = self.tanh(self.hidden3(x3))
        x5 = self.tanh(self.hidden4(x4))
        x6 = self.tanh(self.hidden5(x5))
        out =self.ol(x2)
        return out


mynet = Net()
mynet.load_state_dict(torch.load('BNN_Spectral_Loss_with_tendencyfft_lambda_reg5_PECstep_lead1.pt'))
mynet.eval()
mynet.cuda()

ygrad = torch.zeros([eq_points,input_size,input_size])

for k in range(0,eq_points):

    ygrad [k,:,:] = torch.autograd.functional.jacobian(PECstep,x_torch[k,:])

ygrad = ygrad.detach().cpu().numpy()

print(ygrad.shape)



matfiledata = {}
matfiledata[u'A_matrix_PEC'] = ygrad
hdf5storage.write(matfiledata, '.', path_outputs+'Amatrix_KS_PECstep_tendency_fft_lambda_reg5_11_ICs_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved Predictions')

