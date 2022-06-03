
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
from scipy import signal#нужно для построения Вейвлета
from scipy.spatial import Voronoi, voronoi_plot_2d #для построения диаграммы Вороного
#для картинок
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys #для записи в файл

#вычисление значения  Гауссова ядра в точке х,y
def GaussianKernel(x,y,mu1,mu2,sigma):
    '''(x,y) --- точка в которой вычисляем значение'''
    difference=np.array([x-mu1,y-mu2])#вектор разности (x,y) и  (mu1,mu2)
    distance=np.linalg.norm(difference)#норма вектора разности
    K_sigma=1/(sigma*math.sqrt(2*math.pi))*norm.pdf(distance,loc=0,scale=sigma)
    return K_sigma
     #return 1/(2*math.pi*sigma**2)*math.exp(-((x-mu1)**2+(y-mu2)**2)/(2*sigma**2))

#процудура нахождения плотности в точке (x,y), рассчитаная как сумма Гауссиан по списку точек Samples
def Density(Samples,x,y,sigma):
    '''samples - полученные точки,
     (x,y) - точка, в которой вычисляем плотность;
     sigma --- дисперсия для Гауссиан'''
    d=0#значение плотности
    for i in Samples:
        d+=GaussianKernel(x,y,i[0],i[1],sigma)
    return d/len(Samples)

#рисование поверхности по точкам
def plot_surface(X, Y, Z):
# cтроим график
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

    # диапазон по оси Z:
    minz=np.min(Z)
    maxz=np.max(Z)
    ax.set_zlim(minz-1,maxz+1)

    # настройки осей чтобы было красиво:
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    # определение цвета
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #  сохраняем картинку
    plt.savefig("3d_surface.png")

    # показываем картинку
    plt.show()

    return 0

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

#класс точка диаграммы
class DiagrammPoint:
    def __init__(self):
        self.coordinates=[]#координаты точки
        self.density=0#плотность в этой точке
        self.InGraph=False#true если точка является точкой  из samples
        self.GraphNodeNumber=-1#номер вершины графа, если точка является точкой  из samples
        self.Potential=-1#значение потенциала в

#процедура создания множества точек из исходного графа и точек с диаграммы Вороного
def CreateDiagrammPoints(Samples,sigma):
    DiagrammPoints = []
    k=0
    for i in Samples:
        CurrentDiagrammPoint = DiagrammPoint()
        CurrentDiagrammPoint.coordinates=[i[0],i[1]]
        CurrentDiagrammPoint.density=Density(Samples,i[0],i[1],sigma)
        CurrentDiagrammPoint.InGraph=True
        CurrentDiagrammPoint.GraphNodeNumber=k
        k+=1
        DiagrammPoints.append(CurrentDiagrammPoint)
    vor = Voronoi(Samples)
    #DiagrammPoints=DiagrammPoints+vor.vertices
    for i in vor.vertices:
        CurrentDiagrammPoint = DiagrammPoint()
        CurrentDiagrammPoint.coordinates = [i[0], i[1]]
        CurrentDiagrammPoint.density = Density(Samples, i[0], i[1], sigma)
        CurrentDiagrammPoint.InGraph = False
        DiagrammPoints.append(CurrentDiagrammPoint)
    return DiagrammPoints

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
            l=random.random()/10+0.0001 #сгенерируем маленькое положительное собственное число до 1/10
            i.Potential=(1/l)*i.density
    return GrPt
#выпишем координаты точек сетки и значения потенциала  и плотности в них
def PrintGrid(GrPt):
    original_stdout=sys.stdout
    FileGridPoints=open('GridPointsValue','w')
    sys.stdout=FileGridPoints
    #with open('GridPointsValue','a') as file:
    FileGridPoints.write('x,y,InGraph,Potentia,Density')
    for i in GrPt:
        x=np.around(i.coordinates[0],2)
        y=np.around(i.coordinates[1],2)
        pot=np.around(i.Potential,2)
        dens=np.around(i.density,2)
        #FileGridPoints.write('x'+'y'+'i.InGraph'+'pot'+'dens'+'\n')
        print(x,y,i.InGraph,pot,dens,'\n')
    sys.stdout = original_stdout
    #FileGridPoints.close()

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
#процедура создания массива значений функций potential и density для рисования
#функция фозвращает
            #X - список первых координат точек сетки
            #Y- список вторых координат точек сетки
            #PotentialValue - список значений потенциала
            #DensityValue - список значений плотности
def CreatePotentialAndDensityValues(x0,y0,m,n,h,GrPt):
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Диапазоны по оси X и Y:
    X = np.arange(x0, x0+(m+1)*h, h) # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y0+(n+1)*h, h) # (старт, финиш, шаг бинаризации)

    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)

    #создадим массив значений потенциала, чтобы его нарисовать
    PotentialValue=np.zeros((n+1,m+1))
    DensityValue=np.zeros((n+1,m+1))
    j=0
    for i in GrPt:
        l=math.floor(j/(n+1))
        t=j-l*(n+1)
        #print('l,t',l,t)
        PotentialValue[t,l]=i.Potential
        DensityValue[t,l]=3#max(math.log(i.density),-1)
        j+=1

    return X,Y,PotentialValue,DensityValue
#процедура рисования ланшафта по параметрам сетки и списку точек сетки
def DrawPotentialLandscape(x0,y0,m,n,h,GrPt):
    X,Y,Z,T=CreatePotentialAndDensityValues(x0,y0,m,n,h,GrPt)
    plot_surface(X, Y, Z)
    #  сохраняем картинку
    plt.savefig("PotentialLandscape.png")

def DrawDensityLandscape(x0,y0,m,n,h,GrPt):
    X,Y,Z,T=CreatePotentialAndDensityValues(x0,y0,m,n,h,GrPt)
    plot_surface(X, Y, T)
    #  сохраняем картинку
    #plt.savefig("3d_surface.png")


def sinc(x):
    if x==0:
        s=1
    else:
        s=math.sin(math.pi*x)/(math.pi*x)
    return s
#гладкая функция, принимающая значения grPt.Potential в точках сетки
def SmoothingFunction(x0,y0,m,n,h,Grpt,x,y,d):
    f=0
    for i in Grpt:
        f+=i.Potential*sinc(x-i.coordinates[0])*sinc(y-i.coordinates[1])
    return f



#s=np.loadtxt('StartSamples')#читаю данные из файла как матрицу
s=np.loadtxt('Samples1')#читаю данные из файла как матрицу
print(s)

#s=np.loadtxt('text4.txt',dtype=np.complex_)

df=s
#Create graph from data. knn - Number of nearest neighbors (including self)
G = gt.Graph(df, use_pygsp=True, knn=3)#df - это матрица KxM в которой хранятся первичные вектора.
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
#нарисуем граф
G.set_coordinates(kind=s)
G.plot()

#вычислим параметры сетки и создадим список точек сетки с такими параметрами
m,n,h,x0,y0,sigma=CreateSuitableGridParameters(s)
print('sigma',sigma)
GridPoints=CreateGridPoints(m,n,h,x0,y0,s,sigma)
#вычислим вектор плотностей для точек графа
DensV=CreateSmallDensityVector(s,m,n,h,x0,y0,GridPoints)
print(DensV)

P=PsevdoInverseL_K#псевдообратная матрица к лапласиану графа
#авычислим потенциал во всех точках сетки
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

#Wavelet
from scipy import signal
import matplotlib.pyplot as plt
points=100
a=4
vec2=signal.ricker(points,a)
#plt.plot(vec2)
#plt.show()
#

#построение и изображение диаграммы Вороного
vor=Voronoi(s)
fig=voronoi_plot_2d(vor)
plt.show()
#print('vorVertices',vor.vertices)#вершины диаграммы Вороного

CreateDiagrammPoints(s,1)
#построение графика
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


#DrawPotentialLandscape(x0,y0,m,n,h,GridPoints,'potential')
DrawPotentialLandscape(x0,y0,m,n,h,GridPoints)
DrawDensityLandscape(x0,y0,m,n,h,GridPoints)
#DrawPotentialLandscape(x0,y0,m,n,h,GridPoints,'density')

hh=0.25
#нарисуем гладкую картинку
def DrawSmoothFunction(x0,y0,m,n,h,hh):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Диапазоны по оси X и Y:

    X = np.arange(x0, x0+m*h, hh) # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y0+n*h, hh) # (старт, финиш, шаг бинаризации)

    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    #M=math.floor(n*h/hh)
    #N=math.floor(m*h/hh)
    Z = np.zeros((len(Y),len(X)))

    #print('z',Z)
    for l in range(len(X)):
        for t in range(len(Y)):
            Z[t,l]=SmoothingFunction(x0,y0,m,n,h,GridPoints,x0+l*hh,y0+t*hh,'potential')

    plot_surface(X, Y, Z)
print('h',h)
#DrawSmoothFunction(x0,y0,m,n,h,hh)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Диапазоны по оси X и Y:

X = np.arange(x0, x0+m*h, hh) # (старт, финиш, шаг бинаризации)
Y = np.arange(y0, y0+n*h, hh) # (старт, финиш, шаг бинаризации)

# определяем 2D-сетку
X, Y = np.meshgrid(X, Y)
Z = np.zeros((len(Y),len(X)))

#print('z',Z)
for l in range(len(X)):
    for t in range(len(Y)):
         Z[t,l]=SmoothingFunction(x0,y0,m,n,h,GridPoints,x0+l*hh,y0+t*hh,'potential')
PrintGrid(GridPoints)
plot_surface(X, Y, Z)

#x=np.around(0.3562,decimals=2)
#print(x)