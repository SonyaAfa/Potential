
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
import scipy.spatial.distance as distance
#import statistics
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter

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
        d+=GaussianKernel(x,y,i[0],i[1],sigma)
    return d/len(Samples)
print(Density(s,1,1,0.1))
#процудура создания решетки(сетки)
def CreateSuitableGrid(Samples):
    '''create grid
    (x0,y0) --- left down corner'''
    #находим левый нижний (x0,y0) и правый верхний (x1,y1) узла графа. (x0,y0) --- левый нижний угол сетки
    x0=min(Samples, key=lambda j:j[0])[0]
    x1=max(Samples, key=lambda j:j[0])[0]
    y0=min(Samples, key=lambda j:j[1])[1]
    y1=max(Samples, key=lambda j:j[1])[1]

    MinDist=distance.pdist(Samples).min()#наименьшее расстоние между точками из samples
    h=MinDist/(1.01*math.sqrt(2))# определяю шаг сетки так, чтобы к разным точкам графа
                                 # разные точки сетки оказались ближайшими. Для этого достаточно, чтобы h*sqrt(2)<MinDist
    print('mindist',MinDist,'h',h)
    m=0
    n=0
    while x0+m*h<x1:
        m+=1
    while y0+n*h<y1:
        n+=1
    print('The grid will be', m ,'x', n,'with the step',h,'left corner',(x0,y0))

    return m,n,h,x0,y0
#процедура создания массива точек сетки рпо параметрам сетки
def CreateGridPoints(m,n,h,x0,y0):
    GridPoints = []
    for i in range(m+1):
       for j in range(n+1):
          GridPoints=GridPoints+[[x0+i*h,y0+j*h]]
    return GridPoints

m,n,h,x0,y0=CreateSuitableGrid(s)
print('mnhx0y0x1y1',m,n,h,x0,y0)
print(CreateGridPoints(m,n,h,x0,y0))
#plt.scatter([2,3,1],[0,1,2])
#процедура рисования точек из Samples
def DrawPoints(Samples):
    FirstCoordinate = []
    SecondCoordinate = []
    for i in Samples:
        FirstCoordinate = FirstCoordinate + [i[0]]
        SecondCoordinate = SecondCoordinate + [i[1]]
    print('coord', FirstCoordinate, SecondCoordinate)
    plt.scatter(FirstCoordinate, SecondCoordinate)
    #plt.show()
DrawPoints(s)
GridPoints=CreateGridPoints(m,n,h,x0,y0)
DrawPoints(GridPoints)
plt.show()
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