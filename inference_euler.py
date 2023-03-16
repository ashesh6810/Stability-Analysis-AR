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
from count_trainable_params import count_parameters
import hdf5storage
import pickle



path_outputs = '/home/exouser/RK4_analysis/KS_stuff/new_outputs/'

with open('/home/exouser/RK4_analysis/KS_stuff/models/save/KS.pkl', 'rb') as f:
    data = pickle.load(f)


data=np.asarray(data[:,100000:])


lead=1

inf_fac = 10
time_step = inf_fac*(1e-3)
trainN=80000
input_size = 512
output_size = 512
hidden_layer_size = 1000

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])



def RK4step(net,input_batch):
 output_1 = net(input_batch.cuda())
 output_2 = net(input_batch.cuda()+0.5*output_1)
 output_3 = net(input_batch.cuda()+0.5*output_2)
 output_4 = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1 = net(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1)


def directstep(net,input_batch):
  output_1 = net(input_batch.cuda())
  return output_1

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
        out =self.ol(x6)
        return out


mynet = Net()
count_parameters(mynet)
mynet.cuda()

mynet.load_state_dict(torch.load('BNN_Eulerstep_lead'+str(lead)+'.pt'))



M=20000

pred = np.zeros([M,np.size(label_test,1)])
for k in range(0,M):

    if (k==0):

        out = Eulerstep(mynet,input_test_torch[0,:])
        pred [k,:] = out.detach().cpu().numpy()

    else:

        out = Eulerstep(mynet,torch.from_numpy(pred[k-1,:]).float().cuda())

        pred [k,:] = out.detach().cpu().numpy()

matfiledata = {}
matfiledata[u'prediction'] = pred
matfiledata[u'Truth'] = label_test
hdf5storage.write(matfiledata, '.', path_outputs+'predicted_KS_Eulerstep_inf_fac_'+str(inf_fac)+'_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved Predictions')
