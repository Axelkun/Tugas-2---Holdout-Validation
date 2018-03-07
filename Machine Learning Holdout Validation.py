# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:45:59 2018

@author: Axel
"""
import pylab as pl
import numpy as np
import csv
import random

#Importing data from CSV file
def read_lines():
    with open('Iris.csv') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]
#list of Iris Dataset. x5 is class
x = list(read_lines())

#Input Epoch and Alpha value
n= int(input('enter epoch: '))
a= float(input('enter alpha: '))

#Random Value for Theta 1(q0), Theta 2(q1), Theta 3(q2), Theta 4(q3), and bias
def tq0():
    return random.uniform(-1,1)
def tq1():
    return random.uniform(-1,1)
def tq2():
    return random.uniform(-1,1)
def tq3():
    return random.uniform(-1,1)
def tb():
    return random.uniform(-1,1)


#function for h(x,theta,b)
def h(tq0,tq1,tq2,tq3,tb,i):
    ans=0.0
    ans=(tq0*x[i][0])+(tq1*x[i][1])+(tq2*x[i][2])+(tq3*x[i][3])+tb
    return ans

#function for Sigmoid(h)
def sigmoid(h):
    return 1/(1+np.exp(-1.0*h))

#loss Function
def error(i,s):
    return (s-x[i][4])**2

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
def newq(numb,a,theta):
    return numb+(a*theta)
def newb(b,a,db):
    return b+(a*db)

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

#The MACHINE LEARNING
err=[]
verr=[]
def ML(test,valid):
    q0=tq0();q1=tq1();q2=tq2();q3=tq3();b=tb()
    #q0=0.2;q1=0.2;q2=0.2;q3=0.2;b=0.2
    for i in range (0,n):
        te=0
        for i in range (0,len(test)):
            ha=h(q0,q1,q2,q3,b,i)
            s=sigmoid(ha)
            e=error(i,s)
            te+=e
            dt0=deltaq0(i,s);dt1=deltaq1(i,s);dt2=deltaq2(i,s);dt3=deltaq3(i,s)
            db=deltab(i,s)
            q0=newq(q0,a,dt0);q1=newq(q1,a,dt1);q2=newq(q2,a,dt2);q3=newq(q3,a,dt3)
            b=newb(b,a,db)
        err.append(te/100.0)
        for i in range (0,len(valid)):
            ha=h(q0,q1,q2,q3,b,i)
            s=sigmoid(ha)
            e=error(i,s)
            te+=e
        verr.append(te/100.0)
    return 0

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
ML(training,test)
pl.plot(err,'-r')#RED Line for TRAINING
pl.plot(verr,'-g')#GREEN Line for validation
pl.legend(['training', 'validation'], loc='upper right')

#show the Plot
pl.xlabel('Epoch')
pl.ylabel('Error')
pl.show()
