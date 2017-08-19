import numpy as np
from sklearn.svm import SVC
from sklearn import ensemble
#from sklearn import svm
from sklearn import neighbors
from sklearn import tree
import random as rm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
x=[]
y=[]
Q=[]
x_train=[]
y_train=[]
c=0
neg=[]
pos=[]
f=['e_acid.txt','e_murder.txt','e_cyber.txt','e_terrorist.txt','e_rape.txt','e_dowry.txt','e_theft.txt','e_domestic.txt','e_education.txt','e_social.txt','e_awards.txt','e_science.txt','e_medical.txt']
for i in range(0,len(f)):
    x=[]
    y=[]
    f1=open(f[i],'r')
    data=f1.readlines()
    counter=0
    for j in data:
        counter+=1
        x.append(j)
        if(i<8):
            y.append(0)
        else:
            y.append(1)
        if counter==100:
            break
    for j in range(0,len(x)):
        Q.append((x[j],y[j]))
    

rm.shuffle(Q)
fi=open('test.txt','r')
test_data=fi.readlines()
for i in test_data:
    Q.append((i,0))
t=[0,1,0,0,0,1,1,0,0,1,1,0,0]
for i in range(0,len(Q)):
    x_train.append(Q[i][0])
    y_train.append(Q[i][1])

vec = CountVectorizer()
vec.fit_transform(x_train)
sm = vec.transform(x_train)
sm1=sm.todense()
sm2=sm1[:len(Q)-13]
sm3=sm1[len(Q)-13:]
y1_train=y_train[:-13]
y1_train=np.array(y1_train)
acc=0
rnd=ensemble.RandomForestClassifier()
rnd.fit(sm2,y1_train)
l1=rnd.predict(sm3)

for i in range(0,len(l1)):
    if(l1[i]==0):
        neg.append(test_data[i])
    else:
        pos.append(test_data[i])
        
for i in range(0,len(l1)):
    if l1[i]==t[i]:
        print i
        acc+=1
a=np.array(t)
print acc
met=metrics.classification_report(a,l1)
print met


#######################  MODEL 2   #########################################
####################### negative news ##################################

f=['e_acid.txt','e_murder.txt','e_cyber.txt','e_terrorist.txt','e_rape.txt','e_dowry.txt','e_theft.txt','e_domestic.txt']
c=0
Q1=[]
x1_train=[]
y1_train=[]
for i in range(0,len(f)):
    x=[]
    y=[]
    f1=open(f[i],'r')
    data=f1.readlines()
    counter=0
    for j in data:
        counter+=1
        x.append(j)
        y.append(c)
        if counter==100:
            break
    for j in range(0,len(x)):
        Q1.append((x[j],y[j]))
    c+=1

rm.shuffle(Q1)
for i in neg:
    Q1.append((i,0))
l=len(neg)

for i in range(0,len(Q1)):
    x1_train.append(Q1[i][0])
    y1_train.append(Q1[i][1])

vec = CountVectorizer()
vec.fit_transform(x1_train)
s = vec.transform(x1_train)
s1=s.todense()
s2=s1[:len(Q1)-l]
s3=s1[len(Q1)-l:]
yn_train=y1_train[:-l]
yn_train=np.array(yn_train)

rnd=ensemble.RandomForestClassifier()
rnd.fit(s2,yn_train)
l2=rnd.predict(s3)
for i in range(0,len(neg)):
    print l2[i],'      ',neg[i]
    print '\n'
    
##################### positive  news #################
x2_train=[]
y2_train=[]
f1=['e_education.txt','e_social.txt','e_awards.txt','e_science.txt','e_medical.txt']
c1=0
Q2=[]
for i in range(0,len(f1)):
    x=[]
    y=[]
    f1=open(f[i],'r')
    data=f1.readlines()
    counter=0
    for j in data:
        counter+=1
        x.append(j)
        y.append(c1)
        if counter==100:
            break
    for j in range(0,len(x)):
        Q2.append((x[j],y[j]))
    c1+=1  
    
rm.shuffle(Q2)
for i in pos:
    Q2.append((i,0))
l=len(pos)

for i in range(0,len(Q2)):
    x2_train.append(Q2[i][0])
    y2_train.append(Q2[i][1])

vec = CountVectorizer()
vec.fit_transform(x2_train)
sn = vec.transform(x2_train)
sn1=sn.todense()
sn2=sn1[:len(Q2)-l]
sn3=sn1[len(Q2)-l:]
yn1_train=y2_train[:-l]
yn1_train=np.array(yn1_train)

rnd=ensemble.RandomForestClassifier()
rnd.fit(sn2,yn1_train)
l3=rnd.predict(sn3)
for i in range(0,len(pos)):
    print l3[i],'      ',pos[i]
    print '\n'