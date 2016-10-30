# coding=utf-8

import numpy as np
import random
import math
import matplotlib.pyplot as pt

def sigmod(x):
    if x < -700:
        ans = 0.0
    elif x > 700:
        ans = 1.0
    else:
        ans = 1.0/(1.0 + np.power(np.e, -1*x))
    return ans

def divideLine(beta, x):
    sum = 0.0
    for i in range(0, len(beta)):
        sum += beta[i]*x[i]
    return sum

def calJ(x, y, beta):
    sum = 0.0
    for i in range(0, len(x)):
        h = sigmod(divideLine(beta, x[i]))
        sum += y[i]*math.log(h) + (1-y[i])*math.log(1-h)
    return -1*sum

# 函数功能：计算当前的梯度
def calGradient(theta, x, y):
    gra = []
    for j in range(0, len(theta)):
        sum = 0.0
        for i in range(0, len(x)):
            test = divideLine(theta, x[i])
            # print test
            h = sigmod(test)
            # print 'h',h
            mid = (h-y[i])*x[i][j]
            # print 'mid', mid
            sum += mid
        gra.append(sum)
    # print np.array(gra)
    return np.array(gra)

def updateTheta(theta, gra, a):
    ans=[]
    for i in range(0, len(theta)):
        ans.append(theta[i]-a*gra[i])
    return np.array(ans)

def calNorm(x):
    ans = 0.0
    for i in range(0, len(x)):
        ans += x[i]*x[i]
    return np.power(ans, 0.5)

def gradientDescent(x, y, theta, a, e):
    count=0
    t = theta
    oldGra = gra = calGradient(t, x, y)
    oldTheta = theta
    while True:
        tmp1 = calNorm(gra)
        t = updateTheta(t, gra, a)
        gra = calGradient(t, x, y)
        tmp2 = calNorm(gra)

        if tmp2 > tmp1:
            t = oldTheta
            gra = oldGra
            a = a/2
        else:
            oldTheta = t
            oldGra = gra

        count+=1
        print 'Gradient=',tmp2,', count=',count
        if tmp2<e:
            break

        if count>10000:
            break

    # print t
    return t

def iniTheta(order):
    theta = 200 * np.random.random(size=order + 1) - 100
    return theta

def readData(num):
    x=[]
    y=[]
    f = open("haberman.data")
    line = f.readline()

    count = 0
    while line:

        value = getNum(line)
        tmp = [1.0]
        tmp.append(value[0])
        tmp.append(value[1])
        tmp.append(value[2])
        x.append(tmp)
        # print np.array(tmp)
        y.append(value[3]-1)

        count += 1
        if count == num:
            break;
        line = f.readline()
    f.close()
    return np.array(x), np.array(y)

def getNum(str):
    num = []
    sub = ''
    for j in range(0, len(str)):
        if str[j] >= '0' and str[j] <= '9':
            sub += str[j]
        else:
            num.append(int(sub))
            # print sub
            sub = ''
    # print num
    return np.array(num)

def testData(theta):
    x = []
    y = []
    f = open("haberman.data")
    line = f.readline()
    while line:

        value = getNum(line)
        tmp = [1.0]
        tmp.append(value[0])
        tmp.append(value[1])
        tmp.append(value[2])
        x.append(tmp)
        # print np.array(tmp)
        y.append(value[3] - 1)

        line = f.readline()
    f.close()

    x = np.array(x)
    y = np.array(y)
    estimate = []

    for i in range(0, len(x)):
        tmp =  sigmod(divideLine(theta, x[i]))
        if tmp > 0.5:
            estimate.append(1)
        else:
            estimate.append(0)

    for i in range(0, len(y)):
        if y[i] == estimate[i]:
            print 'Data', i+1, ':', y[i] + 1, 'Estimate:', estimate[i] + 1, 'Y'
        else:
            print 'Data', i+1, ':', y[i] + 1, 'Estimate:', estimate[i] + 1, 'N'




#main---------------------------------------------------
num = 200
e = 1e-3
a= 0.01
order = 3
theta = iniTheta(order)
print theta
#生成数据
x, y = readData(num)
# print x, y
#最速下降法求最优解
new_th = gradientDescent(x, y, theta, a, e)
print new_th

testData(new_th)



