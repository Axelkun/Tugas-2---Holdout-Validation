# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:10:04 2018

@author: Antraxiana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:45:59 2018

@author: Antraxiana
"""
import pylab as pl
import numpy as np
import csv
import random

#Import data from CSV file
def read_lines():
    with open('Iris.csv', 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]
#list of Iris Dataset. x5 is class
x = list(read_lines())

#Input Epoch and Alpha value
n= int(input('enter epoch: '))
a= float(input('enter alpha: '))

#Random Value for Theta 1(q0), Theta 2(q1), Theta 3(q2), Theta 4(q3), and bias
def q0():
    return random.uniform(-1,1)
def q1():
    return random.uniform(-1,1)
def q2():
    return random.uniform(-1,1)
def q3():
    return random.uniform(-1,1)
def b():
    return random.uniform(-1,1)

#function for h(x,theta,b)
def ha(tq0,tq1,tq2,tq3,tb,i):
    return (tq0*x[i][0])+(tq1*x[i][1])+(tq2*x[i][2])+(tq3*x[i][3])+tb

#function for Sigmoid(h)
def sigmoid(h):
    return 1/(1+np.exp(-h))

#Error function
def error(i,s):
    return x[i][4]-s

#loss Function
def loss(e):
    return e**2

#Delta Function for Theta 1(q0), Theta 2(q1), Theta 3(q2), Theta 4(q3), and bias
def deltaq0(i,s):
    return 2*(x[i][4]-s)*(1-s)*s*x[i][0]
def deltaq1(i,s):
    return 2*(x[i][4]-s)*(1-s)*s*x[i][1]
def deltaq2(i,s):
    return 2*(x[i][4]-s)*(1-s)*s*x[i][2]
def deltaq3(i,s):
    return 2*(x[i][4]-s)*(1-s)*s*x[i][3]
def deltab(i,s):
    return 2*(x[i][4]-s)*(1-s)*s*1

#New Value function for Theta 1(q0), Theta 2(q1), Theta 3(q2), Theta 4(q3), and bias
def newq0(tq0,a,dq0):
    return tq0+(a*dq0)
def newq1(tq1,a,dq1):
    return tq1+(a*dq1)
def newq2(tq2,a,dq2):
    return tq2+(a*dq2)
def newq3(tq3,a,dq3):
    return tq3+(a*dq3)
def newb(tb,a,db):
    return tb+(a*db)

#The MACHINE LEARNING
def ML(x):
    ls=[[0.1]*n]*len(x)
    for i in range(0, len(x)):
        tq0=q0();tq1=q1();tq2=q2();tq3=q3();tb=b()
        for j in range (0,n):
            h=ha(tq0,tq1,tq2,tq3,tb,i)
            s=sigmoid(h)
            e=error(i,s)
            ls[i][j]=loss(e)
            dq0=deltaq0(i,s)
            dq1=deltaq1(i,s)
            dq2=deltaq2(i,s)
            dq3=deltaq3(i,s)
            db =deltab(i,s)
            nq0=newq0(tq0,a,dq0)
            nq1=newq1(tq1,a,dq1)
            nq2=newq2(tq2,a,dq2)
            nq3=newq3(tq3,a,dq3)
            nb=newb(tb,a,db)
            tq0=nq0;tq1=nq1;tq2=nq2;tq3=nq3;tb=nb
    ls = np.asarray(ls)
    ML.epoch= np.arange(0, n).reshape(1,n)
    ML.l=ls[0][ML.epoch]
    return 0

#Split Iris dataset into 2 datasets
def sp(x):
    sp.iris1 = sum([x[i:i+50] for i in range(0, len(x),len(x))],[])
    sp.iris2 = sum([x[i:i+50] for i in range(50, len(x), 50)],[])
    return 0

#Split again to get Data for Training and Data for Testing
def ts(ir):
    ts.train = sum([ir[i:i+40] for i in range(0 ,len(ir),len(ir))],[])
    ts.test = sum([ir[i:i+10] for i in range(40, len(ir), 40)],[])
    return 0

#Combine dataset
def com(a,b):
    com.c=a+b
    return com.c

#==============STEP===================
#split the data for training and testing
sp(x)
sp1=sp.iris1
ts(sp1)
tr1=ts.train
ts1=ts.test
sp2=sp.iris2
ts(sp2)
tr2=ts.train
ts2=ts.test
training=com(tr1,tr2)
test=com(ts1,ts2)

#Train the Machine Learing
ML(training)
trEpo=ML.epoch
trls=ML.l
#plot training result
pl.plot(np.array(trEpo.T),np.array(trls.T),'-r')#RED Line for TRAINING

#Test the Machine Learning
ML(test)
tsEpo=ML.epoch
tsls=ML.l
#plot Testing result
pl.plot(np.array(tsEpo.T),np.array(tsls.T),'-g')#GREEN Line for TESTING

#show the Plot
pl.show()