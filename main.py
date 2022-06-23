
import scipy
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
from scipy.spatial import Delaunay #for tesselation and triangulation
import seaborn as sns #for heatmap
#
import mpmath
#from mpmath import*# почему-то на эту строку выдаются ошибки
#для рисования картинок с кружочками
import matplotlib.patches
import matplotlib.path
from matplotlib.lines import Line2D

#класс точка сетки
class GridPoint:
    def __init__(self):
        self.coordinates=[]#координаты точки
        self.density=0#плотность в этой точке
        self.InGraph=False#true если точка является ближайшей точкой сетки для некоторой точки из samples
        self.GraphNodeNumber=-1#номер вершины графа, к которой точка является ближайшей если такая есть
        self.Potential=-1#значение потенциала в этой точке. -1 означает, что потенциал еще не вычислен

#класс точка диаграммы (более разумного разбиения чем сетка)
class DiagrammPoint:
    def __init__(self):
        self.coordinates=[]#координаты точки
        self.density=0#плотность в этой точке
        self.Potential=-1#значение потенциала в этой точке. -1 означает, что потенциал еще не вычислен
        self.radius=0#расстояние до границы области вороного


#вычисление значения  Гауссова ядра в точке х,y
def gaussian_kernel(x,y,mu1,mu2,sigma):
    '''(x,y) --- точка в которой вычисляем значение'''
    difference=np.array([x-mu1,y-mu2])#вектор разности (x,y) и  (mu1,mu2)
    distance=np.linalg.norm(difference)#норма вектора разности
    K_sigma=1/(sigma*math.sqrt(2*math.pi))*norm.pdf(distance,loc=0,scale=sigma)
    return K_sigma
     #return 1/(2*math.pi*sigma**2)*math.exp(-((x-mu1)**2+(y-mu2)**2)/(2*sigma**2))

#процудура нахождения плотности в точке (x,y), рассчитаная как сумма Гауссиан по списку точек Samples
def density(Samples,x,y,sigma):
    '''samples - полученные точки,
     (x,y) - точка, в которой вычисляем плотность;
     sigma --- дисперсия для Гауссиан'''
    d=0#значение плотности
    for i in Samples:
        d+=gaussian_kernel(x,y,i[0],i[1],sigma)
    return d/len(Samples)

#рисование поверхности по точкам
def plot_surface(X, Y, Z,fig,ax):
# cтроим график
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
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

#процедура, рисующая точки samples and GrPt
#GrPt список объектов GridPoints
def draw_grid_and_samples(GrPt,Samples,m,n,h,x0,y0):
    FirstCoordinate = []
    SecondCoordinate = []
    for i in GrPt:
        FirstCoordinate = FirstCoordinate + [i.coordinates[0]]
        SecondCoordinate = SecondCoordinate + [i.coordinates[1]]
    # print('coord', FirstCoordinate, SecondCoordinate)
    plt.scatter(FirstCoordinate, SecondCoordinate,color='black')

    #обведем красным точки, которые возле точек из графа
    for i in Samples:
        x,y,j,t=nearest_grid_point(i[0],i[1],m,n,h,x0,y0)
        plt.scatter([x],[y],color='red')
        #print('gridPointFromSamples',x,y)
    plt.show()

#процудура нахождения подходящих параметров  решетки(сетки)
def create_suitable_grid_parameters(Samples):
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
def nearest_grid_point(x,y,m,n,h,x0,y0):
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

#процедура создания списка точек сетки по параметрам сетки (точка сетки - объект класса GridPoint)
def create_grid_points(m,n,h,x0,y0,Samples,sigma):
    GridPoints = []
    for i in range(m+1):
       for j in range(n+1):
          CurrentGridPoint=GridPoint()
          CurrentGridPoint.coordinates=[x0+i*h,y0+j*h]
          CurrentGridPoint.density=density(Samples,CurrentGridPoint.coordinates[0],CurrentGridPoint.coordinates[1],sigma)
          GridPoints.append(CurrentGridPoint)
    # для каждой точки из samples определяем ближайшую вершину сетки и сообщаем этой вершине сетки, что для нее
    # есть ближайшая из графа
    k=0
    for i in Samples:
        x,y,l,t=nearest_grid_point(i[0],i[1],m,n,h,x0,y0)
        GridPoints[l*(n+1)+t].InGraph=True
        GridPoints[l*(n+1)+t].GraphNodeNumber=k
        GridPoints[l*(n+1)+t].density=density(Samples,i[0],i[1],sigma)#посчитаем в точках близких к графу плотность такой, какая она в точках
                                                                      #графа
        k+=1
    return GridPoints

#класс точка диаграммы (вохможно он не будет нужен)
class DiagrammPoint:
    def __init__(self):
        self.coordinates=[]#координаты точки
        self.density=0#плотность в этой точке
        self.InGraph=False#true если точка является точкой  из samples
        self.GraphNodeNumber=-1#номер вершины графа, если точка является точкой  из samples
        self.Potential=-1#значение потенциала в

#процедура создания множества точек из исходного графа и точек с диаграммы Вороного
def create_diagramm_points(Samples,sigma):
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

#создадим вектор плотностей, соответствующий вершинам сетки,
# которые являются ближайшими к  вершинам графа
# (наверно это можно было сделать раньше и оптимальнне, но пока пусть так)
def create_small_density_vector(Samples, m, n, h, x0, y0,GrPt):
    '''GrPt --- list of GridPoints
    m,n,h,x0,y0 --- grid parameters'''
    DensVector=[]
    for i in Samples:
        x, y, l, t = nearest_grid_point(i[0], i[1], m, n, h, x0, y0)
        DensVector.append(GrPt[l*(n+1)+t].density)# здесь вычисляется плотность не в точке графа, а в ближайшем узле сетки
    return DensVector


#процудура вычисления потенциала в точках графа (независимо от сетки)
def potential_calculation_on_graph(Samples,P,sigma):
    '''P --- psevdo inversre matrix for Laplacian'''
    #вычислим список плотностей в точках графа
    DensVector=[]
    for i in Samples:
        DensVector.append(density(Samples,i[0],i[1],sigma))
    PotentialVector=[]
    k=0
    for i in Samples:
        #вычислим потенциал как P*Dense
        PotentialVector.append(np.dot(P[k],DensVector))
        k+=1
    print('s and potential',Samples)
    print(np.around(DensVector,decimals=2))
    print(np.around(PotentialVector,decimals=2))
    return PotentialVector

#процедура, возвращающая потенциал по формуле Больцмана U=-Dln p(x,y),
# где p - плотность с параметром sigma
def boltzmann_potential(x,y,Samples,sigma,D):
    u=-D*np.log(density(Samples,x,y,sigma))
    return u


#процудура вычисления потенциала во всех точках сетки
#DensVector --- вектор плотностей в точках, ближайших к вершинам графа
def potential_calculation(GrPt,P,DensVector):
    '''P --- psevdo inversre matrix for Laplacian'''
    for i in GrPt:
        #для точек сетки из графа вычислим потенциал как L^{-1}*Dense, для остальных точек l^{-1}*плотность, если
        #плотность небольшая и плотность/100  если плотность >1/100
        if i.InGraph:
            i.Potential=np.dot(P[i.GraphNodeNumber],DensVector)
        else:
            if i.density<0.01:
                l=random.random()/10+0.0001 #сгенерируем маленькое положительное собственное число до 1/10
            else:
                l=1000
            i.Potential=(1/l)*i.density
    return GrPt

#Процедура создающая массивы сетки и значений в ней для того, чтобы можно было нарисовать картинку
def create_potential_and_density_values(x0,y0,m,n,h,GrPt):
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
        DensityValue[t,l]=math.log(i.density)
        j+=1

    return X,Y,PotentialValue,DensityValue

#выпишем координаты точек сетки и значения потенциала  и плотности в них в отдельно созданный файл
def print_grid(GrPt):
    original_stdout=sys.stdout
    FileGridPoints=open('GridPointsValue','w')
    sys.stdout=FileGridPoints
    #with open('GridPointsValue','a') as file:
    FileGridPoints.write('x,y,InGraph,Potential,Density'+'\n')
    for i in GrPt:
        x=np.around(i.coordinates[0],2)
        y=np.around(i.coordinates[1],2)
        pot=np.around(i.Potential,2)
        dens=np.around(i.density,2)
        #FileGridPoints.write('x'+'y'+'i.InGraph'+'pot'+'dens'+'\n')
        print('(',x,y,')',i.InGraph,pot,dens,'\n')
    sys.stdout = original_stdout
    FileGridPoints.close()

#процедура рисования точек из Samples
def draw_points(Samples):
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
            #DensityValue - список значений логарифма от плотности
def create_potential_and_density_values(x0,y0,m,n,h,GrPt):
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
        #DensityValue[t,l]=math.log(i.density)
        DensityValue[t, l] = i.density
        j+=1

    return X,Y,PotentialValue,DensityValue

#процедура рисования ланшафта по параметрам сетки и списку точек сетки
def draw_potentialLandscape(x0,y0,m,n,h,GrPt,fig,ax):
    X,Y,Z,T=create_potential_and_density_values(x0,y0,m,n,h,GrPt)
    plot_surface(X, Y, Z,fig,ax)
    ax = sns.heatmap(Z)
    plt.show()
    #  сохраняем картинку
    #plt.savefig("PotentialLandscape.png")

def draw_densityLandscape(x0,y0,m,n,h,GrPt,fig,ax):
    X,Y,Z,T=create_potential_and_density_values(x0,y0,m,n,h,GrPt)
    plot_surface(X, Y, T,fig,ax)
    ax = sns.heatmap(Z)
    plt.show()
    #  сохраняем картинку
    #plt.savefig("3d_surface.png")


def sinc(x,N):
    if x==0:
        s=1
    else:
        s=math.sin(N*x)/(N*x)
    return s

#
def sinc2d(x,y,N):
    f=sinc((np.linalg.norm([x,y])),N)
    return f

#гладкая функция, принимающая значения grPt.Potential в точках сетки (ее нужно убрать и заменить на следующую)
def smoothing_function(x0,y0,m,n,h,Grpt,x,y,d):
    f=0
    for i in Grpt:
        f+=i.Potential*sinc(x-i.coordinates[0],math.pi)*sinc(y-i.coordinates[1],math.pi)
    return f

#гладкая функция, принимающая значения Values в точках Samples
#Samples - список координат точек
#Values --- список значений
def smoothing_function_sinc(Samples,Values,x,y,N):
    f=0
    for i in range(len(Samples)):
        f+=Values[i]*sinc(x-Samples[i][0],N)*sinc(y-Samples[i][1],N)
    return f

#гладкая функция, принимающая значения Values в точках Samples
#Samples - список координат точек
#Values --- список значений
#процудура сглаживания с учетом радиусов вписанных окружностей в диаграмму Вороного
def smoothing_function_sinc_with_radiuses(Samples,Values,x,y,Radiuses):#тут есть ошибка необнаруженная
    f=0
    for i in range(len(Samples)):
        print('Samples[i][0]',Samples[i][0],Radiuses[i])
        f+=Values[i]*sinc(x-Samples[i][0],math.pi)*sinc(y-Samples[i][1],math.pi)
    return f

#процедурап перечисления вершин графа в том порядке, в котором они встречаются в сетке
def ListingGraphPoints(GrPt):
    for i in GrPt:
        if i.InGraph:
            print(i.GraphNodeNumber)

#нарисуем гладкую картинку
def draw_smooth_function(x0,y0,m,n,h,hh,GrPt,fig,ax):#
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})#почему-то эта строка не выполняется
    # Диапазоны по оси X и Y:
    X = np.arange(x0, x0+m*h, hh) # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y0+n*h, hh) # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    #M=math.floor(n*h/hh)
    #N=math.floor(m*h/hh)

    Z = np.zeros((len(Y),len(X[1])))
    #print('z',Z)
    for l in range(len(X[1])):
        for t in range(len(Y)):
            Z[t,l]=smoothing_function(x0,y0,m,n,h,GrPt,x0+l*hh,y0+t*hh,'potential')
    plot_surface(X, Y, Z,fig,ax)
    ax = sns.heatmap(Z)
    plt.show()

#нарисуем гладкую картинку (без сеток)
def draw_smooth_functiong_general(x0,x1,hx,y0,y1,hy,Samples,Values,N,fig,ax):#
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})#почему-то эта строка не выполняется
    # Диапазоны по оси X и Y:
    X = np.arange(x0, x1, hx) # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y1, hy) # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    #M=math.floor(n*h/hh)
    #N=math.floor(m*h/hh)

    Z = np.zeros((len(Y),len(X[1])))

    for l in range(len(X[1])):
        for t in range(len(Y)):
            #print('l,t',l,t)
            Z[t,l]=smoothing_function_sinc(Samples,Values,x0+l*hx,y0+t*hy,N)

    #print('z',Z)
    plot_surface(X, Y, Z,fig,ax)
    ax=sns.heatmap(Z,center=0,cmap='YlGnBu')
    plt.show()

#процудура, рисующая 3д график функции
def draw_function(x0,y0,x1,y1,h,func,fig,ax):
    X = np.arange(x0, x1, h)  # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y1, h)  # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(Y), len(X[1])))

    for l in range(len(X[1])):
        for t in range(len(Y)):
            Z[t, l] = func(X[t,l],Y[t,l])
    plot_surface(X, Y, Z,fig,ax)

#процудура перечисления матриц графа, записывает их  в файл GraphMatrix
def print_graph_matrix(Samples):
    df = Samples
    # Create graph from data. knn - Number of nearest neighbors (including self)
    G = gt.Graph(df, use_pygsp=True, knn=4)  # df - это матрица KxM в которой хранятся первичные вектора.
    #print('j', G)

    original_stdout=sys.stdout
    FileGraphMatrix=open('GraphMatrix','w')
    sys.stdout=FileGraphMatrix
    #with open('GridPointsValue','a') as file:
    print('matrix'+'\n')

    G.A
    print(' Adjacency matrix: binary version' + '\n')
    print(np.around(G.A, decimals=2))
    #print(' Adjacency matrix' + '\n')
    #print(np.around(G.get_adjacency(), decimals=2))
    print(' The weighted degree' + '\n')
    print(np.around(G.dw, decimals=2))
    print(' Adjacency matrix: K' + '\n')
    print(np.around(G.K, decimals=2))
    # вычислим нормализованный лапласиан графа
    G.compute_laplacian()
    print(' Laplacian' + '\n')
    print(np.around(G.L.A, decimals=2))
    # вычислим нормализованный лапласиан графа и псевдооброатную к нему
    G.compute_laplacian('normalized')
    L_K = G.L.A  # матрица - лапласианг графа
    print(' normalized Laplacian'+'\n')
    print( np.around(L_K, decimals=2))
    PsevdoInverseL_K = LA.pinv(L_K)
    print('PsevdoInverse'+'\n')
    print(np.around(PsevdoInverseL_K, decimals=2))


    sys.stdout = original_stdout
    FileGraphMatrix.close()

#процедура, возвращающая список точек без k-й точки
def delete_point(Samples,k):
    new_list=[]
    j=0
    for i in Samples:
        if j!=k:
            new_list.append(i)
        j+=1
    return new_list

#процудура  рисующая диаграмму Вороного
def draw_voronoi_diagramm(Samples):
    vor=Voronoi(Samples)
    fig=voronoi_plot_2d(vor)
    plt.title('Voronoi')
    plt.show()

#процедура возвращающая список половин от минимального расстояния от точки до остальных точек
#то есть радиус шара наибольшего радиуса с центром в точке, который вписывается в ее окресность Вороного
def half_min_distance(Samples):
    hmd=[]
    k=0
    s=Samples
    for i in Samples:
        dist=scipy.spatial.distance.cdist([s[k]], delete_point(s, k))
        hmd.append(min(min(dist))/2)
        k+=1
    return hmd

#процедура, рисующая точки и окружности, радиуса 1/2 от наименьшего расстояния до остальных точек
def draw_circles(Samples):
    hmd=half_min_distance(Samples)#список 1/2 минимумов расстояний до ближайших точек
    FirstCoordinate = []
    SecondCoordinate = []
    k=0
    s=Samples
    for i in s:
        FirstCoordinate = FirstCoordinate + [i[0]]
        SecondCoordinate = SecondCoordinate + [i[1]]

    fig,ax=plt.subplots()
    axes=plt.gca()
    ax.set_xlim(min(FirstCoordinate)-1,max(FirstCoordinate)+1)
    ax.set_ylim(min(SecondCoordinate)-1,max(SecondCoordinate)+1)
    for i in Samples:
        print('hmdk',hmd[k],i[0])
        circ=plt.Circle((i[0],i[1]),hmd[k],fill=False)
        ax.scatter(i[0],i[1])
        ax.add_patch(circ)
        k+=1

#plt.gca().add_patch(circ1)
    fig.savefig('pltcirc.png')
    plt.show()

def main():
    s=np.loadtxt('Samples1')#читаю данные из файла как матрицу
    print(s)

    print_graph_matrix(s)
    #s=np.loadtxt('text4.txt',dtype=np.complex_)

    df=s
    #Create graph from data. knn - Number of nearest neighbors (including self)
    G = gt.Graph(df, use_pygsp=True, knn=4)#df - это матрица KxM в которой хранятся первичные вектора.

    G.A
    #вычислим нормализованный лапласиан графа и псевдооброатную к нему
    G.compute_laplacian('normalized')
    L_K=G.L.A#матрица - лапласианг графа
    print('Laplacian', L_K)
    PsevdoInverseL_K=LA.pinv(L_K)
    print('P',np.around(PsevdoInverseL_K,decimals=2))

    #нарисуем граф
    G.set_coordinates(kind=s)
    G.plot()

    #вычислим параметры сетки и создадим список точек сетки с такими параметрами
    m,n,h,x0,y0,sigma=create_suitable_grid_parameters(s)
    print('sigma',sigma)


    #создадим список точек сетки используя посчитанные выше параметры
    GridPoints=create_grid_points(m,n,h,x0,y0,s,sigma)
    #вычислим вектор плотностей для точек сетки, являющихся ближайшими к точкам графа
    DensV=create_small_density_vector(s,m,n,h,x0,y0,GridPoints)

    P=PsevdoInverseL_K#псевдообратная матрица к лапласиану графа
    #вычислим потенциал во всех точках сетки
    potential_calculation(GridPoints,P,DensV)

    #Нарисуем сетку и граф
    plt.title('Graph and grid')
    draw_grid_and_samples(GridPoints,s,m,n,h,x0,y0)
    #нарисовали граф и сетку

    #построение ландшафтов (потенциала и плотности)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('PotentialLandscape')
    draw_potentialLandscape(x0,y0,m,n,h,GridPoints,fig,ax)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Density')
    draw_densityLandscape(x0,y0,m,n,h,GridPoints,fig,ax)
    #DrawPotentialLandscape(x0,y0,m,n,h,GridPoints,'density')

    #hh=0.25
    hh=0.25
    print('h',h)
    #нарисуем сглаженный ландшафт потенциала с помощью sinc
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Smooth PotentialLandscape')
    draw_smooth_function(x0,y0,m,n,h,hh,GridPoints,fig,ax)

    #выпишем данные о точках сетки в отдельный файл
    print_grid(GridPoints)
    #перечислим вершины графа (samples) в том порядке, в котором они встречаются в сетке
    ListingGraphPoints(GridPoints)

    #вычислим потенциал в точках графа (не нужно для алгоритма с сетками)
    PotentialVector=potential_calculation_on_graph(s,PsevdoInverseL_K,sigma)
    #вычислили
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Smooth PotentialLandscape from Graph')
    #plt.text(x0, y0+n*h-1,'From_Graph',fontsize=15)
    draw_smooth_functiong_general(x0,x0+n*h,hh,y0,y0+m*h,hh,s,PotentialVector,math.pi,fig,ax)


    #посомотрим что такое потенциал Больцмана
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Boltzmann')
    def b_p(x,y):
        return boltzmann_potential(x,y,s,sigma,1)
    draw_function(x0,y0,x0+m*h,y0+n*h,hh,b_p,fig,ax)


    #построение и изображение диаграммы Вороного
    draw_voronoi_diagramm(s)
    #print('vorVertices',vor.vertices)#вершины диаграммы Вороного
    #триангуляция Делоне
    tri=Delaunay(s)
    #нарисуем результат триангуляции
    plt.title('Delaunay triangulation')
    plt.triplot(s[:,0],s[:,1],tri.simplices)
    plt.plot(s[:,0],s[:,1],'o')
    plt.show()

    print(tri.simplices)

    #посомотрим что такое sinc2d
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Sinc2d')

    def f(x, y):
        return sinc2d(x, y, 3)

    draw_function(-5, -5, 5, 5, hh, f,fig,ax)

    draw_circles(s)
    hmd = half_min_distance(s)
    hmd = np.zeros(len(s))
    for i in range(len(s)):
        hmd[i] = 1
    print('hmd', hmd)

    def g(x, y):
        return smoothing_function_sinc_with_radiuses(s, PotentialVector, x, y, hmd)

    x1 = x0 + m * h
    y1 = y0 + n * h
    X = np.arange(x0, x1, h)  # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y1, h)  # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(Y), len(X[1])))

    for l in range(len(X[1])):
        for t in range(len(Y)):
            Z[t, l] = 1  # smoothing_function_sinc(s,PotentialVector,X[t,l],Y[t,l],math.pi)

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#plt.title('erunda')
#plot_surface(X, Y, Z)
if __name__ == '__main__':
    main()