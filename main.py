
import scipy
import numpy as np
from numpy import linalg as LA

s=np.loadtxt('StartSamples')#читаю данные из файла как матрицу
print(s)
import graphtools as gt
#s=np.loadtxt('text4.txt',dtype=np.complex_)
df=s
#Create graph from data. knn - Number of nearest neighbors (including self)
G = gt.Graph(df, use_pygsp=True, knn=3)
print(G)

#df - это матрица KxM в которой хранятся первичные вектора.
import numpy as np
#import statistics
from scipy.stats import norm
import math
sigma=1
#nn=np.random.normal(mu,sigma,1)
#вычисление значения  Гауссова ядра в точке х,y
def GaussianKernel(x,y,mu1,mu2,sigma):
    '''(x,y) --- точка в которой вычисляем значение'''
    difference=np.array([x-mu1,y-mu2])#вектор разности (x,y) и  (mu1,mu2)
    distance=np.linalg.norm(difference)#норма вектора разности
    K_sigma=1/(sigma*math.sqrt(2*math.pi))*norm.pdf(distance,loc=0,scale=sigma)
    return K_sigma
     #return 1/(2*math.pi*sigma**2)*math.exp(-((x-mu1)**2+(y-mu2)**2)/(2*sigma**2))
print('Gaussian',GaussianKernel(3,4,0,0,1))
print('lenSampl',len(s))
#процудура нахождения плотности в точке (x,y), рассчитаная как сумма Гауссиан по матрице точек Samples
def Density(Samples,x,y,sigma):
    '''samples - полученные точки,
     (x,y) - точка, в которой вычисляем плотность;
     sigma --- дисперсия для Гауссиан'''
    d=0#значение плотности
    for i in Samples:
        #print((x, y, i[0], i[1], sigma))
        #print('GaussianKernel',GaussianKernel(x,y,i[0],i[1],sigma))
        d+=GaussianKernel(x,y,i[0],i[1],sigma)
        #print('d',d)
        #print('density',i,i[0])
    return d/len(Samples)
print(Density(s,1,1,0.1))
#процудура создания решетки mxn c шагом h
def CreateGrid():
    '''create grid'''
    m=int(input('Enter the wide of the grid'))
    n=int(input('Enter the height of the grid'))
    h=float(input('enter the side of one square'))#шаг сетки
    x0, y0=eval(input('enter the coordinate of the bottom left corner'))
    print('The grid will be', m ,'x', n,'with the step',h,'left corner',(x0,y0))
    return m,n,h,(x0,y0)
#m,n,h,(x0,y0)=CreateGrid()
#print(m,n,h)
#процудура нахождения ближайшей точки решетки в случае если точка расположена внутри области, покрытой решеткой
def NearestGridPoint(x,y,m,n,h,x0,y0):
    '''(x,y) --- the point
    m,n - wide and height of the grid
    h - size of squares
    (x0,y0) --- left corner of the grid'''
    x_nearest='out of grid'
    y_nearest='out of grid'
    if ((x<x0) or (x>x0+m*h) or (y<y0) or (y>y0+n*h)):
        print('this point is out of grid')
    else:
        l=round((x-x0)/h)
        t=round((y-y0)/h)
        x_nearest=x0+l*h
        y_nearest=y0+t*h
    return x_nearest,y_nearest
#x=float(input('enter x'))
#y=float(input('enter y'))
#print(NearestGridPoint(x,y,m,n,h,x0,y0))
print('new commit')