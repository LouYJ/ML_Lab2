# coding=utf-8

import numpy as np
import random
import math
import matplotlib.pyplot as pt

def sigmod(x):
    ans = 1.0/(1.0 + np.power(np.e, -1*x))
    return ans

def genrateData(num):
    mu = 0
    sigma = 1.5

    x_up = 3*np.random.random(num)
    x_down = 3*np.random.random(num)
    y_up = []
    y_down = []
    for i in range(0,num):
        offset_Guassian = random.gauss(mu, sigma)
        y_up.append(4.0-x_up[i]+abs(offset_Guassian))
        offset_Guassian = random.gauss(mu, sigma)
        y_down.append(2.0-x_down[i]-abs(offset_Guassian))

    return x_up, y_up, x_down, y_down

def divideLine(beta, x):
    sum = 0.0
    for i in range(0, len(beta)):
        sum += beta[i]*x[i]
    return sum

def calJ(x, y, beta, lda):
    sum = 0.0
    for i in range(0, len(x)):
        h = sigmod(divideLine(beta, x[i]))
        sum += y[i]*math.log(h) + (1-y[i])*math.log(1-h)
    norm = calNorm(theta)
    sum += lda*norm*norm #加入惩罚项
    return -1*sum

# 函数功能：计算当前的梯度
def calGradient(theta, x, y, lda):
    gra = []
    for j in range(0, len(theta)):
        sum = 0.0
        for i in range(0, len(x)):
            h = sigmod(divideLine(theta, x[i]))
            # print 'h',h
            mid = (h-y[i])*x[i][j]
            sum += mid
        sum += lda*theta[j] #加入惩罚项
        gra.append(sum)
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

def gradientDescent(x, y, theta, a, e, lda):
    count=0
    t = theta
    while True:
        # print t
        gra = calGradient(t, x, y, lda)
        # print gra
        t = updateTheta(t, gra, a)
        tmp = calNorm(gra)
        count+=1
        print 'Gradient=',tmp,', count=',count
        if tmp<e:
            break

        if count>10000:
            break

    print t
    return t

def mergeX(x_up, y_up, x_down, y_down, beta):
    x=[]
    y=[]
    for i in range(0, len(x_up)):
        ele=[]
        ele.append(1.0)
        ele.append(x_up[i])
        ele.append(y_up[i])
        x.append(ele)
        y.append(1)
    for i in range(0, len(x_down)):
        ele = []
        ele.append(1.0)
        ele.append(x_down[i])
        ele.append(y_down[i])
        x.append(ele)
        y.append(0)
    return np.array(x), np.array(y)

def iniTheta(order):
    theta = 200 * np.random.random(size=order + 1) - 100
    return theta

#main---------------------------------------------------
num = 15
e = 1e-3
a= 0.01
order = 2
lda = 0.1
theta = iniTheta(order)
print theta
#生成数据
x_up, y_up, x_down, y_down = genrateData(num)
x, y = mergeX(x_up, y_up, x_down, y_down, theta)
#最速下降法求最优解
new_th = gradientDescent(x, y, theta, a, e, lda)

#画图----------------------------------------------------
fig = pt.figure()
ax = fig.add_subplot(111)

base_x = np.arange(0, 3, 0.001)
base_y = 3.0-base_x
ans_y = -1*(new_th[0]+new_th[1]*base_x)/new_th[2]

ax.plot(base_x,ans_y,color='green',linestyle='-',marker='')
ax.plot(base_x,base_y,color='black',linestyle='-',marker='')
ax.plot(x_up,y_up,'bo')
ax.plot(x_down,y_down,'ro')
ax.legend()

pt.show()
