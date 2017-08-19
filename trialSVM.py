import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import tree
import random as rm
import matplotlib.pyplot as pl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
def precisionRecall(met):
    ll1=[]
    ll=[]
    s=''
    k=''
    ll2=[]
    ll3=[]
    ll4=[]
    l5=[]
    ll=met.split(' ')
    for i in ll:
        if(i!=''):
            ll1.append(i)
    ll1.remove(ll1[0])
    ll1.remove(ll1[0])
    ll1.remove(ll1[0])
    ll1.remove(ll1[0])
    for j in ll1:
        s=j
        k=''
        for j in s:
            if (j>='0' and j<='9') or (j=='.'):
                k=k+j
        if(k!=''):
            ll2.append(k)
    ll2.append(0)
    c=0
    for i in ll2:
        if(c>=5):
            ll4.append(ll3)
            ll3=[]
            c=0
        if(c<5):
            a=float(i)
            ll3.append(a)
            c=c+1
    for i in ll4:
        i=i[:-2]
        l5.append(i)
    return l5
lines=[]
lines1=[]
x=[]
y=[]
Q=[]
x_train=[]
y_train=[]
count=0
c=0
c1=0
c2=0
Q1=[]
neg=[]
pos=[]
xnew_train=[]
xnew_test=[]
ynew_train=[]
ynew_test=[]
y_model=[]
y1_model=[]
new_neg=[]
new_pos=[]
f=['e_acid.txt','e_murder.txt','e_cyber.txt','e_terrorist.txt','e_rape.txt','e_dowry.txt','e_theft.txt','e_domestic.txt','e_education.txt','e_social.txt','e_awards.txt','e_science.txt','e_medical.txt']
for i in range(0,len(f)):
    x=[]
    y=[]
    f1=open(f[i],'r')
    data=f1.readlines()
    for j in data:
        x.append(j)
        if(i<8):
            y.append((0,count))
        else:
            y.append((1,count))
    xt,xtt,yt,ytt=train_test_split(x,y,test_size=0.40)
    for j in range(0,len(xt)):
        Q.append((xt[j],yt[j]))
    for j in range(0,len(xtt)):
        Q1.append((xtt[j],ytt[j]))
    count+=1
rm.shuffle(Q)
Q=Q[:6000]
for i in  range(0,len(Q1)):
    Q.append(Q1[i])
for i in range(0,len(Q)):
    x_train.append(Q[i][0])
    y_train.append(Q[i][1][0])


vec = CountVectorizer()
vec.fit_transform(x_train)
sm = vec.transform(x_train)
sm1=sm.todense()
sm2=sm1[:len(x_train)-len(Q1)]
sm3=sm1[len(x_train)-len(Q1):]
y1_train=y_train[:-len(Q1)]
y1_train=np.array(y1_train)
acc=0

clf=svm.SVC()
clf.fit(sm2,y1_train)
l1=clf.predict(sm3)
accu=clf.score(sm2,y1_train)
print accu
for i in range(0,len(l1)):
    if l1[i]==Q1[i][1][0]:
        if(l1[i]==0):
            new_neg.append(Q1[i])
        else:
            new_pos.append(Q1[i])
        acc+=1
a=[]
for i in range(0,len(Q1)):
    a.append(Q1[i][1][0])
a=np.array(a)
#print acc
met=metrics.classification_report(a,l1)
print met

lines=precisionRecall(met)

for label,precision,recall in lines:
    pl.plot([recall, precision],label=label)
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall')
pl.legend(loc="upper right")
pl.show()
'''
#######################  MODEL 2   #########################################
####################### negative news ##################################

f=['e_acid.txt','e_murder.txt','e_cyber.txt','e_terrorist.txt','e_rape.txt','e_dowry.txt','e_theft.txt','e_domestic.txt']
c=0
x_train=[]
y_train=[]
Q=[]
for i in range(0,len(f)):
    x=[]
    y=[]
    f1=open(f[i],'r')
    data=f1.readlines()
    for j in data:
        x.append(j)
        y.append((0,c))
    xt,xtt,yt,ytt=train_test_split(x,y,test_size=0.20)
    for j in range(0,len(xt)):
        Q.append((xt[j],yt[j]))
    c+=1

rm.shuffle(Q)
Q=Q[:6000]
for i in new_neg:
    Q.append(i)
for i in range(0,len(Q)):
    x_train.append(Q[i][0])
    y_train.append(Q[i][1][1])

vec = CountVectorizer()
vec.fit_transform(x_train)
sm=vec.transform(x_train)
sm1=sm.todense()
sm2=sm1[:len(x_train)-len(new_neg)]
sm3=sm1[len(x_train)-len(new_neg):]
y1_train=y_train[:-len(new_neg)]
y1_train=np.array(y1_train)

clf=svm.SVC()
clf.fit(sm2,y1_train)
l1=clf.predict(sm3)
acc2=clf.score(sm2,y1_train)
print acc2
acc=0
for i in range(0,len(l1)):
    if l1[i]==new_neg[i][1][1]:
        acc+=1
a=[]
for i in range(0,len(new_neg)):
    a.append(new_neg[i][1][1])
a=np.array(a)
print acc
met=metrics.classification_report(a,l1)
print met

##################### positive  news #################

f=['e_education.txt','e_social.txt','e_awards.txt','e_science.txt','e_medical.txt']
c=8
x_train=[]
y_train=[]
Q=[]
for i in range(0,len(f)):
    x=[]
    y=[]
    f1=open(f[i],'r')
    data=f1.readlines()
    for j in data:
        x.append(j)
        y.append((1,c))
    xt,xtt,yt,ytt=train_test_split(x,y,test_size=0.20)
    for j in range(0,len(xt)):
        Q.append((xt[j],yt[j]))
    c+=1

rm.shuffle(Q)
Q=Q[:6000]
for i in new_pos:
    Q.append(i)
for i in range(0,len(Q)):
    x_train.append(Q[i][0])
    y_train.append(Q[i][1][1])

vec = CountVectorizer()
vec.fit_transform(x_train)
sm=vec.transform(x_train)
sm1=sm.todense()
sm2=sm1[:len(x_train)-len(new_pos)]
sm3=sm1[len(x_train)-len(new_pos):]
y1_train=y_train[:-len(new_pos)]
y1_train=np.array(y1_train)

clf=svm.SVC()
clf.fit(sm2,y1_train)
l1=clf.predict(sm3)


acc=0
for i in range(0,len(l1)):
    if l1[i]==new_pos[i][1][1]:
        acc+=1
a=[]
for i in range(0,len(new_pos)):
    a.append(new_pos[i][1][1])
a=np.array(a)
print acc
met=metrics.classification_report(a,l1)
print met
'''