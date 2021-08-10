#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:56:13 2018

@author: ljx
"""

#simulations on DeepMTT

#==============================================================================
#1--------DeepMTT network
#==============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

import time
#from batchdata_derive3 import*
from maxout import max_out as MO     #maxout activation function
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

def noisy_af(Xt):
    #noisy activation function
    p = 1
    c = 1
    h = tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5) -0.5
#    h = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    y = h + c*tf.square(K.sigmoid(p*(h-Xt))-0.5) * tf.random.normal(tf.shape(Xt),mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
    return y

get_custom_objects().update({'noisy_af': noisy_af})

def piecewise(Xt):                  #piecewise activation function
#    y = (tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5)-0.5) #piecewise activation function
    y = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    return y

def fir_filter(x,w,b_size,t_size): #快速FIR滤波
    with tf.name_scope('FIR_filter'):
        #x,待滤波的数据,shape为[batch_size,time_size,output_size]
        #w,滤波网络,shape为[fir_size, output_size]
#        shape_x = x.get_shape().as_list()
        shape_w = w.get_shape().as_list()
        x_add = tf.constant(tf.zeros((b_size[0],shape_w[0]-1,shape_w[1])), dtype=tf.float32, shape=[b_size[0],shape_w[0]-1,shape_w[1]], name='X_add')
        x = tf.concat([x_add,x],1)  #给前面不足长度的待滤波序列补全零
        x = tf.expand_dims(x,2)     #扩展一维，存储待滤波数据
        y = []
        for i in range(shape_w[0]):
            y.append(x[:,i:i+t_size,:,:])
        z = tf.concat(y,2)
        z = z*w
        return tf.reduce_sum(z,axis=2)

#数据处理2-----batch里面每一个数据按照第一个值的最大值进行归一化
def Data_Pro2(data):
    weight = np.max(np.abs(data[:,0,:]), axis=1)
    weight = np.transpose(np.array([[weight]]),[2,1,0])
    results = data/weight
    return results, weight
#==============================================================================
#==============================================================================
#建立lstm网络

#超参数的定义-----------------------------------------
# Set RNN parameter
#BN = 0.2
#lr = 1e-5
#lr = 1e3
#处理数据的batch大小
bs = 16
_batch_size = np.array([int(bs)])
# The size of batch for learning，这是在图里的batch_size张量定义
batch_size = tf.constant(value=16, dtype=tf.int32)
# 每个时刻的输入特征是4维的，就是每个时刻输入一行，一行有距离x,y和速度vx,vy
input_size = 4
# 时序持续长度
timestep_size = 64

## 隐含层的数量
#hidden_size = 64
# LSTM layer 的层数
layer_num = 3
#第一层隐层的节点数
hidden_size_1 = 128
#第二层隐层的节点数
hidden_size_2 = 256
#第三层隐层的节点数
hidden_size_3 = 256
#输出层maxout节点数
maxout_size = 64
#正则项系数
lambda1 = 0.003
#FIR滤波层滤波阶数
fir_size = 5

# 最后输出向量的维度
output_size = 4

keep_prob = tf.constant(0, dtype=tf.float32)


#输入输出定义------------------------------------------------------------------
X = tf.Variable(initial_value=tf.zeros(shape=(batch_size, timestep_size, input_size)), shape=(batch_size, timestep_size, input_size), dtype = tf.float32, name='input_x', trainable=False)
y = tf.Variable(initial_value=tf.zeros(shape=(batch_size, timestep_size, output_size)), shape=(batch_size, timestep_size, output_size), dtype = tf.float32, name='input_y', trainable=False)
Xtrac = tf.Variable(initial_value=tf.zeros(shape=(batch_size, timestep_size, input_size)), shape=(batch_size, timestep_size, input_size), dtype=tf.float32, name='input_Xtrac', trainable=False)
myd = tf.Variable(initial_value=tf.zeros(shape=(batch_size, timestep_size-1, output_size)), shape=(batch_size, timestep_size-1, output_size), dtype=tf.float32, name='input_myd', trainable=False)
#FIR
Fir_w = tf.Variable(tf.constant(1.0/fir_size,dtype=tf.float32,shape=[fir_size, input_size]), dtype=tf.float32, name='fir_w')
tf.summary.histogram('fir_w',Fir_w)
X_f = fir_filter(X, Fir_w, _batch_size, timestep_size)
print(X_f.shape)



from tensorflow.keras.layers import LSTMCell
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import RNN
from tensorflow.keras.layers import Input


class Maxout(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)

    def build(self, input_shape):
        self.hidden_size = input_shape[2]
        self.timestep_size = input_shape[1]
        self.maxout_w = self.add_weight(shape=(self.hidden_size, timestep_size, self.hidden_size),
                                        initializer=self.initializer,
                                        name="maxout_w",
                                        trainable=True,
                                        regularizer = tf.keras.regularizers.l2(l2=lambda1))
        self.maxout_b = self.add_weight(name='maxout_b',
                                        shape=(timestep_size, self.hidden_size),
                                        initializer=tf.keras.initializers.Constant(0.1),#, shape=[timestep_size, self.hidden_size]),
                                        trainable=True)

        self.W = self.add_weight(shape=(maxout_size, timestep_size, output_size),
                                 initializer=self.initializer,
                                 name="output_w",
                                 trainable=True,
                                 regularizer = tf.keras.regularizers.l2(l2=lambda1))
        self.b = self.add_weight(name='output_b',
                                 shape=(timestep_size, output_size),
                                 initializer=tf.keras.initializers.Constant(0.1),#tf.constant(0.1, shape=[timestep_size, output_size]),
                                 trainable=True)



    def call(self, inputs):
        T_o = tf.transpose(inputs, [1, 0, 2])
        T_w = tf.transpose(self.maxout_w, [1, 0, 2])
        maxout_o = tf.transpose(tf.matmul(T_o, T_w), [1, 0, 2]) + self.maxout_b
        mo_f = MO(maxout_o, maxout_size, axis=2)  # maxout

        tran_o = tf.transpose(mo_f, [1, 0, 2])
        tran_w = tf.transpose(self.W, [1, 0, 2])
        y_pre = tf.transpose(tf.matmul(tran_o, tran_w), [1, 0, 2]) + self.b

        lstm_outputs = {'maxout_w': self.maxout_w, 'maxout_b': self.maxout_b, 'output_w': self.W, 'output_b': self.b}  # 把要保存的参数做成字典

        return y_pre

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Maxout, self).get_config()
        config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))

from tensorflow_addons.layers import Maxout as Maxout_layer
#构建lstm网络，多层分开--------------------------------------------------------
def lstm_model(batch_size, timestep_size, input_size):

    input_layer = Input(name='the_input', shape=(timestep_size, input_size), batch_size=batch_size)
    lstm_cell_1_fw = LSTMCell(units=hidden_size_1, unit_forget_bias=1.0, dropout=0, recurrent_dropout=keep_prob, activation="noisy_af")
    # lstm_cell_1_fw = tf.nn.RNNCellDropoutWrapper(cell=lstm_cell_1_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell_1_fw.get_initial_state(tf.zeros(shape=(batch_size, timestep_size, input_size)), batch_size, dtype=tf.float32)
    outputs_l1_fw = RNN(lstm_cell_1_fw, return_sequences=True, time_major=False)

    lstm_cell_1_bw = LSTMCell(units=hidden_size_1, unit_forget_bias=1.0, dropout=0, recurrent_dropout=keep_prob, activation="noisy_af")
    # lstm_cell_1_bw = tf.nn.RNNCellDropoutWrapper(cell=lstm_cell_1_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell_1_bw.get_initial_state(tf.zeros(shape=(batch_size, timestep_size, input_size)), batch_size, dtype=tf.float32)
    outputs_l1_bw = RNN(lstm_cell_1_bw, go_backwards=True, return_sequences=True, time_major=False)

    outputs_l1 = Bidirectional(outputs_l1_fw, backward_layer=outputs_l1_bw, merge_mode="concat")(input_layer)

    #forward:
    lstm_cell_2_fw = LSTMCell(units=hidden_size_2, unit_forget_bias=1.0, dropout=0, recurrent_dropout=keep_prob, activation="noisy_af")
    # lstm_cell_2_fw = tf.nn.RNNCellDropoutWrapper(cell=lstm_cell_2_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell_2_fw.get_initial_state(tf.zeros(shape=(batch_size, timestep_size, hidden_size_2)), batch_size, dtype=tf.float32)
    outputs_l2_fw = RNN(lstm_cell_2_fw, return_sequences=True,time_major=False)
    #backward:
    lstm_cell_2_bw = LSTMCell(units=hidden_size_2, unit_forget_bias=1.0, dropout=0, recurrent_dropout=keep_prob, activation="noisy_af")
    # lstm_cell_2_bw = tf.nn.RNNCellDropoutWrapper(cell=lstm_cell_2_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell_2_bw.get_initial_state(tf.zeros(shape=(batch_size, timestep_size, hidden_size_2)), batch_size, dtype=tf.float32)
    outputs_l2_bw = RNN(lstm_cell_2_bw, return_sequences=True, go_backwards=True, time_major=False)

    #output:
    outputs_l2 = Bidirectional(outputs_l2_fw, backward_layer=outputs_l2_bw, merge_mode="concat")(outputs_l1)

    #forward:
    lstm_cell_3_fw = LSTMCell(units=hidden_size_3, unit_forget_bias=1.0)
    lstm_cell_3_fw.get_initial_state(tf.zeros(shape=(batch_size, timestep_size, hidden_size_3)), batch_size, dtype=tf.float32)
    outputs_l3_fw = RNN(lstm_cell_3_fw, return_sequences=True, time_major=False)
    #backward:
    lstm_cell_3_bw = LSTMCell(units=hidden_size_3, unit_forget_bias=1.0)
    lstm_cell_3_bw.get_initial_state(tf.zeros(shape=(batch_size, timestep_size, hidden_size_3)), batch_size, dtype=tf.float32)
    outputs_l3_bw = RNN(lstm_cell_3_bw, go_backwards=True, return_sequences=True, time_major=False)
    #output:
    outputs_l3 = Bidirectional(outputs_l3_fw, backward_layer=outputs_l3_bw, merge_mode="concat")(outputs_l2)
    outputs = outputs_l3
    hidden_size = hidden_size_3*2

    y_pre = Maxout(input_shape=outputs.shape)(outputs)#Maxout_layer(num_units=input_size, axis=2)(outputs)#
    y_final = y_pre + Xtrac
    y_delta = y_final[:, 1:, :] - y_final[:, :-1, :]

    model = tf.keras.Model(inputs=input_layer, outputs=[y_final, y_delta])
    print(model.summary())
    return model

    
    #网络参数提取
    # lstm_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=lstm_net.name)

    
# Set CPU/GPF mode
#------------------------------------------------------------------------------

#==============================================================================
#模型加载
model = lstm_model(batch_size, timestep_size, input_size)
# model_path = "/home/ljx/文档/OpenSources/DeepMTT/Models/LMTT.ckpt"
# load_path = saver.restore(sess, model_path)
# print ("Model restored from file: %s" % model_path)
#
#
# #==============================================================================
# #2--------Trajectory setting
# #==============================================================================
# #==============================================================================
#==============================================================================
from numpy.linalg import cholesky
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints as SP
#采样时刻
sT = 0.1
# 1.匀速直线运动，状态转移方程
F_cv = np.array([[1,0,sT,0],[0,1,0,sT],[0,0,1,0],[0,0,0,1]],'float64')
F_cv = np.transpose(F_cv,[1,0])
def F_ct(w):
    m = np.array([[1,0,np.sin(w*sT)/w,(np.cos(w*sT)-1)/w],
                  [0,1,-(np.cos(w*sT)-1)/w,np.sin(w*sT)/w],
                  [0,0,np.cos(w*sT),-np.sin(w*sT)],
                  [0,0,np.sin(w*sT),np.cos(w*sT)]],'float64')
    return np.transpose(m,[1,0])
#生成航迹函数
def trajectory_creat(data_len, T_matrix, X_a):
    data = np.array([[0 for i in range(4)] for j in range(data_len)],'float64')
    for i in range(data_len):
        data[i,:] = X_a
        X_a = np.dot(X_a, T_matrix)
    data_n = data + np.dot(np.random.randn(data_len, 4), chol_var)  #加噪声
    return data, data_n
#
# #跟踪航迹建模, Parameter setting
# ##T1-----------------
# #d_x = -18000   #目标x方位
# #d_y = 2000   #目标y方位
# #v_x = 150     #目标x方向速度
# #v_y = 200    #目标x方向速度
# #
# #n1=300
# #F1 = F_cv
# #n2=400
# #w1 = 9.18*np.pi/180
# #F2 = F_ct(w1)
# #n3=300
# #w2 = -3.54*np.pi/180
# #F3 = F_ct(w2)
#
# ##T2-----------------
# #d_x = -7000   #目标x方位
# #d_y = -24000   #目标y方位
# #v_x = 280     #目标x方向速度
# #v_y = 320    #目标x方向速度
# #
# #n1=400
# #w1 = -2.08*np.pi/180
# #F1 = F_ct(w1)
# #n2=200
# #F2 = F_cv
# #n3=400
# #w2 = 5.34*np.pi/180
# #F3 = F_ct(w2)
#
# #T3-----------------
# #d_x = 12000   #目标x方位
# #d_y = 13000   #目标y方位
# #v_x = 230     #目标x方向速度
# #v_y = 190    #目标x方向速度
#
# #n1=300
# #F1 = F_cv
# #n2=400
# #w1 = -7.16*np.pi/180
# #F2 = F_ct(w1)
# #n3=300
# #w2 = 4.24*np.pi/180
# #F3 = F_ct(w2)
#
# ##T4-----------------
# #d_x = 5000   #目标x方位
# #d_y = -5000   #目标y方位
# #v_x = 10     #目标x方向速度
# #v_y = 330    #目标x方向速度
# #
# #n1 = 200
# #F1 = F_cv
# #n2 = 600
# #w2 = 3.26*np.pi/180
# #F2 = F_ct(w2)
# #n3 = 200
# #F3 = F_cv
#
# ##T5-----------------
# #d_x = 25000   #目标x方位
# #d_y = -6000   #目标y方位
# #v_x = 120     #目标x方向速度
# #v_y = 230    #目标x方向速度
# #
# #n1 = 220
# #F1 = F_cv
# #n2 = 560
# #w2 = 7.16*np.pi/180
# #F2 = F_ct(w2)
# #n3 = 220
# #F3 = F_cv
#
# ##T6-----------------
# #d_x = 20000   #目标x方位
# #d_y = -20000   #目标y方位
# #v_x = -220     #目标x方向速度
# #v_y = -200    #目标x方向速度
# #
# #n1 = 600
# #w1 = -0.58*np.pi/180
# #F1 = F_ct(w1)
# #n2 = 100
# #F2 = F_cv
# #n3 = 300
# #w3 = -2.21*np.pi/180
# #F3 = F_ct(w3)
#
# ##T7-----------------
# #d_x = -15000   #目标x方位
# #d_y = -25000   #目标y方位
# #v_x = 100     #目标x方向速度
# #v_y = 280    #目标x方向速度
# #
# #n1 = 600
# #w1 = 0.17*np.pi/180
# #F1 = F_ct(w1)
# #n2 = 300
# #F2 = F_cv
# #n3 = 100
# #w3 = -9.19*np.pi/180
# #F3 = F_ct(w3)
#
# ##T8-----------------
# #d_x = -25000   #目标x方位
# #d_y = -15000   #目标y方位
# #v_x = -120     #目标x方向速度
# #v_y = 200    #目标x方向速度
# #
# #n1 = 300
# #w1 = -6.18*np.pi/180
# #F1 = F_ct(w1)
# #n2 = 500
# #w2 = 8.33*np.pi/180
# #F2 = F_ct(w2)
# #n3 = 200
# #w3 = -2.21*np.pi/180
# #F3 = F_ct(w3)
#
# ##T9-----------------
# #d_x = -30000   #目标x方位
# #d_y = -5000   #目标y方位
# #v_x = 250     #目标x方向速度
# #v_y = 180    #目标x方向速度
# #
# #n1 = 550
# #w1 = -1.15*np.pi/180
# #F1 = F_ct(w1)
# #n2 = 150
# #w2 = 9.13*np.pi/180
# #F2 = F_ct(w2)
# #n3 = 300
# #F3 = F_cv
#
#T0-----------------
d_x = -10000   #目标x方位
d_y = 25000   #目标y方位
v_x = 220     #目标x方向速度
v_y = 213    #目标x方向速度

n1 = 400
w1 = -3.38*np.pi/180
F1 = F_ct(w1)
n2 = 200
w2 = 6.82*np.pi/180
F2 = F_ct(w2)
n3 = 400
w3 = -1.17*np.pi/180
F3 = F_ct(w3)

#Beginning Point
x0 = np.array([[d_x, d_y, v_x, v_y]],'float64')
#运动噪声（加速度噪声m/s^2）
state_n = 10.0
s_var = np.square(state_n)
T2 = np.power(sT,2)
T3 = np.power(sT,3)
T4 = np.power(sT,4)
#var_m = np.array([[T4/4,0,T3/2,0],[0,T4/4,0,T3/2],[T3/2,0,T2,0],[0,T3/2,0,T2]],'float64') * s_var
var_m = np.array([[T4/4,0,0,0],[0,T4/4,0,0],[0,0,T2,0],[0,0,0,T2]],'float64') * s_var
chol_var = cholesky(var_m)

#观测噪声
dis_n = 10.0   #距离
dis_var = np.square(dis_n)
azi_n = 8.0     #方位角
azi_var = np.square(azi_n/1000)


num_trajectory = 1000
#状态转移函数
def my_fx(x, sT):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1,0,sT,0],[0,1,0,sT],[0,0,1,0],[0,0,0,1]],'float64') #F_cv

    return np.dot(F, x)
#观测函数
def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r = np.array([0,0],'float64')
    r[0] = np.arctan2(x[1],x[0])
    r[1] = np.sqrt(np.square(x[0])+np.square(x[1]))
    return r
#观测时间
my_t = np.arange(0, sT*(num_trajectory-1), sT)
#定义一个跟踪函数，生成跟踪轨迹
def UKF_tracking(x0, Obser):
    #定义UKF
    dim_state = len(x0)
    [data_len, dim_obser] = Obser.shape
    my_SP = SP(dim_state,kappa=0.)
    my_UKF = UKF(dim_x=dim_state, dim_z=dim_obser, dt=sT, hx=my_hx, fx=my_fx, points=my_SP)
    my_UKF.Q *= var_m
    my_UKF.R *= np.array([[azi_var,0],[0,dis_var]])
    my_UKF.x = x0
    my_UKF.P *= 1.
    #跟踪的航迹结果
    xs = []
    xs.append(my_UKF.x.copy())
    for i in range(data_len-1):
        my_UKF.predict()
        my_UKF.update(Obser[i+1,:])
        xs.append(my_UKF.x.copy())
    return np.asarray(xs)
#把航迹封装成batch
def batch(batch_size, x0, Obser):
    [data_len, dim_obser] = Obser.shape
    Traj_s = np.array([[[0 for i in range(4)] for j in range(data_len)] for k in range(batch_size)],'float64')
    for i in range(batch_size):#每条航迹及其跟踪结果的生成
        xs = UKF_tracking(x0, Obser)
        Traj_s[i] = np.asarray(xs)
    return Traj_s

#==============================================================================
#3-----Tracking with DeepMTT, Monte Carlo Simulation
#==============================================================================
#record data
MC_number=1
org_tj_all = np.array([[[0 for i in range(4)] for j in range(1000)] for k in range(MC_number)],'float64')
observation_all = np.array([[[0 for i in range(2)] for j in range(1000)] for k in range(MC_number)],'float64')
dmtt_tr_all = np.array([[[0 for i in range(4)] for j in range(1000)] for k in range(MC_number)],'float64')
pos_noise = 20
vel_noise = 2


for ii in range(MC_number):
    flag = 1
    while flag == 1:
        try:
            x0_noise = np.array([np.random.normal(0,pos_noise,2),np.random.normal(0,vel_noise,2)])
            org_tj1, dtn1 = trajectory_creat(n1, F1, x0+np.reshape(x0_noise,[4,]))
            org_tj2, dtn2 = trajectory_creat(n2, F2, org_tj1[-1,:])
            org_tj3, dtn3 = trajectory_creat(n3, F3, org_tj2[-1,:])
            org_tj = np.vstack((org_tj1,org_tj2,org_tj3))
            dtn = np.vstack((dtn1,dtn2,dtn3))
            org_tj_all[ii,:,:]=org_tj

            #我的观测
            observation = np.array([[0 for i in range(2)] for j in range(num_trajectory)],'float64')
            observation[:,0] = np.arctan2(dtn[:,1],dtn[:,0]) + np.random.normal(0,azi_var,num_trajectory) #方位角
            observation[:,1] = np.sqrt(np.square(dtn[:,0])+np.square(dtn[:,1])) + np.random.normal(0,dis_var,num_trajectory)   #距离
            observation_all[ii,:,:]=observation

            #设计一个迭代的间隔
            deta_iter = 10
            #LSTM网络输出的次数
            iter_num = int(num_trajectory / deta_iter)

            #通过lstm网络修正跟踪结果
            xs=[]
            b_p = 0
            xb = org_tj[b_p,:]
            isbegin = 1
            is_clac_t = 1
            start_t = time.clock()
            for i in range(iter_num-4):
#                if is_clac_t == 1:
#                    start_t = time.clock()

                xi = xb
                obser_seg = observation[b_p:b_p+timestep_size,:]
                Traj_s = batch(bs, xi, obser_seg)
                my_inputs,_ = Data_Pro2(Traj_s)
                Traj_deta = sess.run(y_pre, feed_dict={X: my_inputs, keep_prob: 1.0, batch_size: _batch_size})
                Traj_c = Traj_s + Traj_deta
                b_p = b_p+deta_iter
                if isbegin == 1:
                    Traj_ult = Traj_c
                    isbegin = 0
                    Traj_pre = Traj_ult
                else:
                    Traj_ult = Traj_c
                    Traj_ult[:,0:timestep_size-deta_iter,:] = 0.5*(Traj_ult[:,0:timestep_size-deta_iter,:]+Traj_pre[:,deta_iter:,:])
                    Traj_pre = Traj_ult
                xb = Traj_ult[0,deta_iter,:]
                xs.append(Traj_ult[0,0:deta_iter,:])
#                print ("step %d has been calculated"% i)

#                if is_clac_t == 1:
#                    end_t = time.clock()
#                    is_clac_t = 0
            end_t = time.clock()
            xs_a = np.asarray(xs)
            xs_b = np.reshape(xs_a,[960,4])
            dmtt_tr_all[ii,:,:] = np.vstack((xs_b,Traj_ult[0,deta_iter:,:]))
            print ("%d MC runs have been calculated"% ii)
            flag = 0
        except:
            print ("Tracking Error in the %d th MC run" % ii)
#==============================================================================
##保存成mat数据
#import scipy.io as scio
#mydata = {'org_tj_all':org_tj_all, 'observation_all':observation_all, 'dmtt_tr_all':dmtt_tr_all, 'my_t':my_t}
#scio.savemat('Tracking10_all', mydata)

import matplotlib.pyplot as plt
plt.figure(1) # 创建图表1
plt.plot(org_tj[:,0], org_tj[:,1])
plt.plot(dmtt_tr_all[0,:,0], dmtt_tr_all[0,:,1])
