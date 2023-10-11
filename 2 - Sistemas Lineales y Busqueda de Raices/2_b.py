from matplotlib import pyplot as plt
import numpy as np

def jacobi_ponderado(A, k, b, N=25, xn=None, x0 = None, tol=1e-10):
    #Chequea que la matriz sea de norma infinito >= 1
    if np.linalg.norm(A,ord=np.inf) < 1:
        return None
    # Creamos una aproximación inicial nula si no se proporciona
    if xn is None:
        xn = np.zeros(len(A))
    if x0 is None:
        x0 = np.zeros(len(A))
    # Creamos una matriz diagonal D y una matriz R que contiene los coeficientes no diagonales
    D = np.diag(A)
    R = A - np.diagflat(D)
    iteracion = 1
    # Iteramos hasta que se alcance una solución aproximada aceptable o se llegue al número máximo de iteraciones
    for i in range(N):
        iteracion = i
        # Calculamos la siguiente aproximación
        x_ponderado = (k+1)*xn - k*x0
        x_new = (b - np.dot(R, x_ponderado)) / D
        # Verificamos la convergencia del método
        if np.linalg.norm(x_new - xn) < tol:
            return x_new, iteracion
        # Actualizamos la aproximación anterior
        x0 = xn
        xn = x_new       
    return xn, iteracion

def jacobi(A, b, N=1000, x0=None, tol=1e-10):
    if x0 is None:
        x0 = np.zeros(len(A))
    D = np.diag(A)
    R = A - np.diagflat(D)
    iteracion = 1
    for i in range(N):
        iteracion = i
        x_new = (b - np.dot(R, x0)) / D
        if np.linalg.norm(x_new - x0) < tol:
            return x_new, iteracion
        x0 = x_new
    return x0, iteracion

def gauss_seidel(A, k, b, N=25, xn=None, x0 = None, tol=1e-10):
    if xn is None:
        xn = np.zeros(len(A))
    if x0 is None:
        x0 = np.zeros(len(A))
    iteracion = 0
    for i in range(N):
        iteracion = i
        x_ponderado = (k+1)*xn - k*x0
        x_prev = x_ponderado.copy()
        for j in range(len(x_ponderado)):
            x_ponderado[j] = (b[j] - np.dot(A[j, :j], x_ponderado[:j]) - np.dot(A[j, j+1:], x_prev[j+1:])) / A[j, j]
        if np.linalg.norm(x_ponderado - x_prev) < tol:
            return x_ponderado, iteracion 
        x0 = xn
        xn = x_ponderado
    return x_ponderado, iteracion

A =  np.loadtxt("2matriz.txt", dtype=float, delimiter=" ")
b = np.loadtxt("2b.txt", dtype=float, delimiter=" ")

#Valores de K (-1,1)
rangoK = np.arange(-0.99,1.01,0.01)

iteraciones_por_K_ponderado = []
iteraciones_por_K_gs = []
iteraciones_por_K = []
[sol,it] = jacobi(A, b, N=1000, x0 = None, tol=1e-10)
for i in rangoK:
    [sol_ponderado,it_ponderado] = jacobi_ponderado(A, i, b, N=1000, xn=None, x0 = None,tol=1e-10)
    [sol_gauss_seidel, it_gauss_seidel] = gauss_seidel(A, i, b, N=1000, xn=None,x0=None, tol=1e-10)
    iteraciones_por_K.append(it) #La cantidad de iteraciones del metodo original sera siempre la misma
    iteraciones_por_K_gs.append(it_gauss_seidel)
    iteraciones_por_K_ponderado.append(it_ponderado)

# Ahora graficamos los resultados
plt.figure(figsize=(10, 6))
plt.plot(rangoK, iteraciones_por_K_gs, label='Gauss-Seidel Ponderado')
plt.plot(rangoK, iteraciones_por_K_ponderado, label='Jacobi Ponderado')
plt.plot(rangoK, iteraciones_por_K, label='Jacobi')
plt.xlabel('Valor de K')
plt.ylabel('Iteraciones')
plt.title('Iteraciones vs Valor de K para los metodos')
plt.grid(True)
plt.legend()
plt.show()