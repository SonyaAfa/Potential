
import scipy
import numpy as np
from numpy import linalg as LA
import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import random
from sklearn import datasets#новая строка из документации к graphtools
import graphtools as gt


#вычисление значения  Гауссова ядра в точке х,y
def GaussianKernel(x,y,mu1,mu2,sigma):
    '''(x,y) --- точка в которой вычисляем значение'''
    difference=np.array([x-mu1,y-mu2])#вектор разности (x,y) и  (mu1,mu2)
    distance=np.linalg.norm(difference)#норма вектора разности
    K_sigma=1/(sigma*math.sqrt(2*math.pi))*norm.pdf(distance,loc=0,scale=sigma)
    return K_sigma
     #return 1/(2*math.pi*sigma**2)*math.exp(-((x-mu1)**2+(y-mu2)**2)/(2*sigma**2))

#процудура нахождения плотности в точке (x,y), рассчитаная как сумма Гауссиан по матрице точек Samples
def Density(Samples,x,y,sigma):
    '''samples - полученные точки,
     (x,y) - точка, в которой вычисляем плотность;
     sigma --- дисперсия для Гауссиан'''
    d=0#значение плотности
    for i in Samples:
        d+=GaussianKernel(x,y,i[0],i[1],sigma)
    return d/len(Samples)

#процудура нахождения подходящих параметров  решетки(сетки)
def CreateSuitableGridParameters(Samples):
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
    #print('mindist',MinDist,'h',h)
    m=0
    n=0
    while x0+m*h<x1:
        m+=1
    while y0+n*h<y1:
        n+=1
    print('The grid will be', m ,'x', n,'with the step',h,'left corner',(x0,y0))
    sigma=h/3#стоит подумать каким именно его лучше выбирать...
    return m,n,h,x0,y0,sigma

#в гриде получится (m+1)*(n+1) точек

#процудура нахождения координат и расположения ближайшей точки решетки в
# случае если точка расположена внутри области, покрытой решеткой
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
    return x_nearest,y_nearest,l,t
#параметры l,t указывают на место этой точки в сетке

#класс точка сетки
class GridPoint:
    def __init__(self):
        self.coordinates=[]#координаты точки
        self.density=0#плотность в этой точке
        self.InGraph=False#true если точка является ближайшей точкой сетки для некоторой точки из samples
        self.GraphNodeNumber=-1#номер вершины графа, к которой точка является ближайшей если такая есть
        self.Potential=-1#значение потенциала в этой точке. -1 означает, что потенциал еще не вычислен

#процедура создания списка точек сетки по параметрам сетки (точка сетки - объект класса GridPoint)
def CreateGridPoints(m,n,h,x0,y0,Samples,sigma):
    GridPoints = []
    for i in range(m+1):
       for j in range(n+1):
          CurrentGridPoint=GridPoint()
          CurrentGridPoint.coordinates=[x0+i*h,y0+j*h]
          CurrentGridPoint.density=Density(Samples,CurrentGridPoint.coordinates[0],CurrentGridPoint.coordinates[1],sigma)
          GridPoints.append(CurrentGridPoint)
    # для каждой точки из samples определяем ближайшую вершину сетки и сообщаем этой вершине сетки, что для нее
    # есть ближайшая из графа
    k=0
    for i in Samples:
        x,y,l,t=NearestGridPoint(i[0],i[1],m,n,h,x0,y0)
        GridPoints[l*(n+1)+t].InGraph=True
        GridPoints[l*(n+1)+t].GraphNodeNumber=k
        k+=1
    return GridPoints


#ШАГИ 4 и 6 Вычисление значений потенциала в узлах решетки

#создадим вектор плотностей в вершинах графа (наверно это можно было сделать раньше и оптимальнне, но пока пусть так)
def CreateSmallDensityVector(Samples, m, n, h, x0, y0,GrPt):
    '''GrPt --- list of GridPoints
    m,n,h,x0,y0 --- grid parameters'''
    DensVector=[]
    for i in Samples:
        x, y, l, t = NearestGridPoint(i[0], i[1], m, n, h, x0, y0)
        DensVector.append(GrPt[l*(n+1)+t].density)# здесь вычисляется плотность не в точке графа, а в ближайшем узле сетки
    return DensVector


#процудура вычисления потенциала во всех точках сетки
def PotentialCalculation(GrPt,P,DensVector):
    '''P --- psevdo inversre matrix for Laplacian'''
    for i in GrPt:
        #для точек сетки из графа вычислим потенциал как L^{-1}*P, для остальных точек l^{-1}*плотность
        if i.InGraph:
            i.Potential=np.dot(P[i.GraphNodeNumber],DensVector)
        else:
            l=random.random()/10+0.0000001 #сгенерируем маленькое положительное собственное число до 1/10
            i.Potential=(1/l)*i.density
    return GrPt

#ШАГ7


#s=np.loadtxt('StartSamples')#читаю данные из файла как матрицу
s=np.loadtxt('Samples1')#читаю данные из файла как матрицу
print(s)

#s=np.loadtxt('text4.txt',dtype=np.complex_)

df=s
#Create graph from data. knn - Number of nearest neighbors (including self)
G = gt.Graph(df, use_pygsp=True, knn=2)
print('j',G)
G.A
#вычислим нормализованный лапласиан графа и псевдооброатную к нему
G.compute_laplacian('normalized')
L_K=G.L.A#матрица - лапласианг графа
print(L_K)
PsevdoInverseL_K=LA.pinv(L_K)
#print(PsevdoInverseL_K)
#print(np.dot(PsevdoInverseL_K,L_K))


#print(G.A)
#print(G.K)
#print('dw',G.dw)
#print('e',G.e)
#G.compute_fourier_basis
G.set_coordinates(kind=s)
G.plot()

#df - это матрица KxM в которой хранятся первичные вектора.

m,n,h,x0,y0,sigma=CreateSuitableGridParameters(s)
print('sigma',sigma)
GridPoints=CreateGridPoints(m,n,h,x0,y0,s,sigma)

DensV=CreateSmallDensityVector(s,m,n,h,x0,y0,GridPoints)
print(DensV)

P=PsevdoInverseL_K
PotentialCalculation(GridPoints,P,DensV)
#выпишем значения потенциала
for l in range(m+1):
    for t in range(n+1):
         print(GridPoints[l*(n+1)+t].Potential,GridPoints[l*(n+1)+t].InGraph,GridPoints[l*(n+1)+t].density)



#перечислим в каком порядке вершины графа встречаются в сетке
for i in GridPoints:
    if i.InGraph:
        print(i.GraphNodeNumber)


#Нарисуем сетку и граф
#print('mnhx0y0x1y1',m,n,h,x0,y0)
#print(CreateGridPoints(m,n,h,x0,y0))

#процедура рисования точек из Samples
def DrawPoints(Samples):
    FirstCoordinate = []
    SecondCoordinate = []
    for i in Samples:
        FirstCoordinate = FirstCoordinate + [i[0]]
        SecondCoordinate = SecondCoordinate + [i[1]]
    #print('coord', FirstCoordinate, SecondCoordinate)
    plt.scatter(FirstCoordinate, SecondCoordinate)
    #plt.show()
#yарисуем граф и сетку
DrawPoints(s)

#DrawPoints(GridPoints)
FirstCoordinate = []
SecondCoordinate = []
for i in GridPoints:
    FirstCoordinate = FirstCoordinate + [i.coordinates[0]]
    SecondCoordinate = SecondCoordinate + [i.coordinates[1]]
# print('coord', FirstCoordinate, SecondCoordinate)
plt.scatter(FirstCoordinate, SecondCoordinate)

#обведем красным точки, которые возле точек из графа

for i in s:
    x,y,j,t=NearestGridPoint(i[0],i[1],m,n,h,x0,y0)
    plt.scatter([x],[y],color='red')
    #print('gridPointFromSamples',x,y)
plt.show()
#нарисовали граф и сетку


#x=float(input('enter x'))
#y=float(input('enter y'))
#print(NearestGridPoint(x,y,m,n,h,x0,y0))
