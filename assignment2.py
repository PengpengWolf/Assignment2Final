# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:53:04 2020

@author: irisy
"""


import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import random

import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
from pandapower.plotting.plotly import simple_plotly
import mykmeans as mkm
import myknn as mknn
def timeseries(output_dir,state='NS'):
    # 1. create test net
    net = create_net()

    # 2. create (random) data source
    n_timesteps = 60
    profiles, ds = create_data_source(n_timesteps)
    #print(ds)
    #print(profiles)
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds)
    #print(net)
    #print(ds)
    #print(profiles)
    #pp.plotting.simple_plot(net)
    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)
   # print(time_steps)
    
    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps)

def create_net(state='NS'):
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init_vm_pu = "flat", init_va_degree = "dc", calculate_voltage_angles=True)
       
    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    b4 = pp.create_bus(net, 110)
    b5 = pp.create_bus(net, 110)
    b6 = pp.create_bus(net, 110)
    b7 = pp.create_bus(net, 110)
    b8 = pp.create_bus(net, 110)
    
    pp.create_ext_grid(net, b0)
    
    pp.create_line(net, b0, b3, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b3, b4, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b3, b8, 10, "149-AL1/24-ST1A 110.0")
    if state in ['NS','HL','LL','GD']:
        pp.create_line(net, b4, b5, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b5, b2, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b8, b7, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b5, b6, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b1, b7, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b7, b6, 10, "149-AL1/24-ST1A 110.0")
    
    if state == 'HL':
        pp.create_load(net, b4, p_mw=90*1.1, q_mvar=30*1.1, name='load1')
        pp.create_load(net, b6, p_mw=100*1.1, q_mvar=35*1.1, name='load2')
        pp.create_load(net, b8, p_mw=125*1.1, q_mvar=50*1.1, name='load3')
    
    if state == 'LL':
        pp.create_load(net, b4, p_mw=90*0.9, q_mvar=30*0.9, name='load1')
        pp.create_load(net, b6, p_mw=100*0.9, q_mvar=35*0.9, name='load2')
        pp.create_load(net, b8, p_mw=125*0.9, q_mvar=50*0.9, name='load3')   
    else: 
        pp.create_load(net, b4, p_mw=90, q_mvar=30, name='load1')
        pp.create_load(net, b6, p_mw=100, q_mvar=35, name='load2')
        pp.create_load(net, b8, p_mw=125, q_mvar=50, name='load3')
        
    pp.create_gen(net, b0, p_mw=0, vm_pu=1.0, name='gen1', slack=True)    
    pp.create_sgen(net, b1, p_mw=163, q_mvar=0, name='sgen1')
    if state in ['NS','HL','LL','LD']:
        pp.create_sgen(net, b2, p_mw=85, q_mvar=0, name='sgen2')
    
    return net

def create_data_source(n_timesteps=60,state='NS'):
    profiles = pd.DataFrame()
    if state=='HL':
        profiles['load1_p'] = np.random.random(n_timesteps) * 90*1.1
        profiles['load2_p'] = np.random.random(n_timesteps) * 100*1.1
        profiles['load3_p'] = np.random.random(n_timesteps) * 125*1.1
    if state=='LL':
        profiles['load1_p'] = np.random.random(n_timesteps) * 90*0.9
        profiles['load2_p'] = np.random.random(n_timesteps) * 100*0.9
        profiles['load3_p'] = np.random.random(n_timesteps) * 125*0.9        
    if state=='NS':
        profiles['load1_p'] = np.random.random(n_timesteps) * 90
        profiles['load2_p'] = np.random.random(n_timesteps) * 100
        profiles['load3_p'] = np.random.random(n_timesteps) * 125        
        
        profiles['sgen1_p'] = np.random.random(n_timesteps) * 163
    if state in ['NS','HL','LL','LD']:
        profiles['sgen2_p'] = np.random.random(n_timesteps) * 85 
  
    ds = DFData(profiles)
    return profiles, ds

def create_controllers(net, ds,state='NS'):
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["load1_p"])
    ConstControl(net, element='load', variable='p_mw', element_index=[1],
                 data_source=ds, profile_name=["load2_p"])
    ConstControl(net, element='load', variable='p_mw', element_index=[2],
                 data_source=ds, profile_name=["load3_p"])    
    ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["sgen1_p"])
    if state in ['NS','HL','LL','LD']:
        ConstControl(net, element='sgen', variable='p_mw', element_index=[1],
                    data_source=ds, profile_name=["sgen2_p"])

    
def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    #ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
   # ow.log_variable('res_line', 'loading_percent')
    #ow.log_variable('res_line', 'i_ka')
    return ow
output_dir_HL = os.path.join(tempfile.gettempdir(), "time_series_example_HL")
output_dir_LL = os.path.join(tempfile.gettempdir(), "time_series_example_LL")
output_dir_GD = os.path.join(tempfile.gettempdir(), "time_series_example_GD")
output_dir_LD = os.path.join(tempfile.gettempdir(), "time_series_example_LD")
output_dir_NS = os.path.join(tempfile.gettempdir(), "time_series_example_NS")

#output_dir_NS = os.path.join(tempfile.gettempdir(), "time_series_vm")
print("Results can be found in your local temp folder: {}".format(output_dir_HL))
if not os.path.exists(output_dir_HL):
    os.mkdir(output_dir_HL)
timeseries(output_dir_HL,state='HL')
timeseries(output_dir_LL,state='LL')
timeseries(output_dir_GD,state='GD')
timeseries(output_dir_LD,state='LD')
timeseries(output_dir_NS,state='NS')

vm_pu_file = os.path.join(output_dir_HL, "res_bus", "vm_pu.xls")
vm_pu_1 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_HL, "res_bus", "va_degree.xls")
angle_deg_1 = pd.read_excel(angle_deg_file)

vm_pu_file = os.path.join(output_dir_LL, "res_bus", "vm_pu.xls")
vm_pu_2 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_LL, "res_bus", "va_degree.xls")
angle_deg_2 = pd.read_excel(angle_deg_file)
temp_vm1=np.vstack((vm_pu_1,vm_pu_2))
temp_ang1=np.vstack((angle_deg_1,angle_deg_2))

vm_pu_file = os.path.join(output_dir_GD, "res_bus", "vm_pu.xls")
vm_pu_3 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_GD, "res_bus", "va_degree.xls")
angle_deg_3 = pd.read_excel(angle_deg_file)
temp_vm2=np.vstack((temp_vm1,vm_pu_3))
temp_ang2=np.vstack((temp_ang1,angle_deg_3))

vm_pu_file = os.path.join(output_dir_LD, "res_bus", "vm_pu.xls")
vm_pu_4 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_LD, "res_bus", "va_degree.xls")
angle_deg_4 = pd.read_excel(angle_deg_file)
temp_vm3=np.vstack((temp_vm2,vm_pu_4))
temp_ang3=np.vstack((temp_ang2,angle_deg_4))

vm_pu_file = os.path.join(output_dir_NS, "res_bus", "vm_pu.xls")
vm_pu_5 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_NS, "res_bus", "va_degree.xls")
angle_deg_5 = pd.read_excel(angle_deg_file)
vm_pu_tot=np.vstack((temp_vm3,vm_pu_5))
angle_deg_tot=np.vstack((temp_ang3,angle_deg_5))

###K-means
bus_num=9
status=5
data_bus=[]
dataset=[]
dataset_18D=[]


for i in range(bus_num):
    data_temp=[]
    data_bus.append([vm_pu_tot[:,i+1],angle_deg_tot[:,i+1]])
    bus_var=np.var(data_bus[i][1])#
    if bus_var !=0:
        #data_bus[i][1]/=bus_var
        data_bus[i][1]=data_bus[i][1]
    else:
        data_bus[i][1]=data_bus[i][1]
    for j in range(len(data_bus[i][0])):
        data_temp.append([data_bus[i][0][j],data_bus[i][1][j]])
    dataset.append(data_temp)
for i in range(len(dataset[0])):
    data_temp=[]
    for j in range(bus_num):
        if j!=0:
            data_temp.append(dataset[j][i])
    dataset_18D.append(data_temp)
#data_bus.append([vm_pu_tot[:,bus_num],angle_deg_tot[:,bus_num]])
#bus_var=np.var(data_bus[0][1])#calcilate the variation of the angle value
#data_bus[0][1]/=bus_var#normalize the angle value
#for i in range(len(data_bus[0][0])):#reshape the values,so that every element contains one vol and its corresponding angle.
#    dataset.append([data_bus[0][0][i],data_bus[0][1][i]])

#dim = np.shape(dataset)
n_timesteps=60
k=5#innitailize cluter number
maxiter=100
#toler=0.001
centroids=[]
centroid=[]
clusterAssments=[]
clusterAssment=[]
dataset_tags=[]
dataset_tag=[]
data_final=[]

centroid, clusterAssment=mkm.kmeans(dataset_18D,k,maxiter)
#centroids.append(centroid)
#clusterAssments.append(clusterAssment)
dataset_tags=mkm.showCluster(dataset_18D, k, centroid, clusterAssment)
#dataset_tags.append(dataset_tag)
#centroids, clusterAssment=mkm.kmeans(dataset,k,maxiter)
#dataset_tag=mkm.showCluster(dataset, k, centroids, clusterAssment)
#plt.title('Kmeans')
num_HL=[]
num_HL=mkm.countTag(dataset_tags[0:n_timesteps],k)
num_LL=[]
num_LL=mkm.countTag(dataset_tags[n_timesteps:2*n_timesteps],k)
num_GD=[]
num_GD=mkm.countTag(dataset_tags[2*n_timesteps:3*n_timesteps],k)
num_LD=[]
num_LD=mkm.countTag(dataset_tags[3*n_timesteps:4*n_timesteps],k)
num_NS=[]
num_NS=mkm.countTag(dataset_tags[4*n_timesteps:5*n_timesteps],k)
num=0
num=num_HL[0]+num_LL[1]+num_GD[2]+num_LD[3]+num_NS[4]
tot_num=5*n_timesteps*9
#print('Rate of Effiency:','\n','Total:',num/tot_num,'\n','HL: ',num_HL[0]/tot_num*5,'\n','LL: ', num_LL[0]/tot_num*5,'\n','GD: ',num_GD[0]/tot_num*5,'\n','LD: ',num_LD[0]/tot_num*5,'\n','NS: ',num_NS[0]/tot_num*5)
###KNN

c=int(0.78*n_timesteps)


#training_HL, training_LL,training_GD,training_LD,training_NS, 
#test_HL, test_LL, test_GD, test_LD, test_NS

training_HL, test_HL=mknn.seperateTT(dataset_tags[0:n_timesteps],c,n_timesteps)
print('The data number in the training set is :', len(training_HL*9))
print('The data number in the test set is :', len(test_HL*9))
test_num=8#initailize cluter number
#vote for HL status
#index_HL=[]
non_correct_HL=0
yes_correct_HL=0
dis_HL=mknn.getdistance(test_HL,training_HL)
dis_tag_HL,non_correct_HL,yes_correct_HL=mknn.compareResult(dis_HL,test_num)
print('Non correct for HL state:',non_correct_HL,'\n','Correct for HL state:',yes_correct_HL,'\n','Rate of accuracy for HL state:',yes_correct_HL/(len(test_HL)))

training_LL, test_LL=mknn.seperateTT(dataset_tags[n_timesteps:2*n_timesteps],c,n_timesteps)
print('The data number in the training set is :', len(training_LL*9))
print('The data number in the test set is :', len(test_LL*9))
non_correct_LL=0
yes_correct_LL=0
dis_LL=mknn.getdistance(test_LL,training_LL)
dis_tag_LL,non_correct_LL,yes_correct_LL=mknn.compareResult(dis_LL,test_num)
print('Non correct for LL state:',non_correct_LL*9,'\n','Correct for LL state:',yes_correct_LL*9,'\n','Rate of accuracy for LL state:',yes_correct_LL/(len(test_LL)))

training_GD, test_GD=mknn.seperateTT(dataset_tags[2*n_timesteps:3*n_timesteps],c,n_timesteps)
print('The data number in the training set is :', len(training_GD*9))
print('The data number in the test set is :', len(test_GD*9))
non_correct_GD=0
yes_correct_GD=0
dis_GD=mknn.getdistance(test_GD,training_GD)
dis_tag_GD,non_correct_GD,yes_correct_GD=mknn.compareResult(dis_GD,test_num)
print('Non correct for GD state:',non_correct_GD*9,'\n','Correct for GD state:',yes_correct_GD*9,'\n','Rate of accuracy for GD state:',yes_correct_GD/(len(test_GD)))

training_LD, test_LD=mknn.seperateTT(dataset_tags[3*n_timesteps:4*n_timesteps],c,n_timesteps)
print('The data number in the training set is :', len(training_LD*9))
print('The data number in the test set is :', len(test_LD*9))
non_correct_LD=0
yes_correct_LD=0
dis_LD=mknn.getdistance(test_LD,training_LD)
dis_tag_LD,non_correct_LD,yes_correct_LD=mknn.compareResult(dis_LD,test_num)
print('Non correct for LD state:',non_correct_LD*9,'\n','Correct for LD state:',yes_correct_LD*9,'\n','Rate of accuracy for LD state:',yes_correct_LD/(len(test_LD)))

training_NS, test_NS=mknn.seperateTT(dataset_tags[4*n_timesteps:5*n_timesteps],c,n_timesteps)
print('The data number in the training set is :', len(training_NS*9))
print('The data number in the test set is :', len(test_NS*9))
non_correct_NS=0
yes_correct_NS=0
dis_NS=mknn.getdistance(test_NS,training_NS)
dis_tag_NS,non_correct_NS,yes_correct_NS=mknn.compareResult(dis_NS,test_num)
print('Non correct for NS state:',non_correct_NS*9,'\n','Correct for NS state:',yes_correct_NS*9,'\n','Rate of accuracy for NS state:',yes_correct_NS/(len(test_NS)))

non_correct=non_correct_HL+non_correct_LL+non_correct_LD+non_correct_GD+non_correct_NS
yes_correct=yes_correct_HL+yes_correct_LL+yes_correct_LD+yes_correct_GD+yes_correct_NS

print('\n','Non correct for all state:',non_correct*9,'\n','Correct for all state:',yes_correct*9,'\n','Rate of accuracy for all state:',yes_correct/(len(test_HL+test_LL+test_GD+test_LD+test_NS)))

