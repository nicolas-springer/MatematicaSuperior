import time
import numpy as np
from scipy.sparse.linalg import cg 
import scipy

def conjugate_gradient(A, b, x0, tol=1e-20):
    # Utiliza la función de Gradiente Conjugado de SciPy para resolver el sistema
    num_iters = 0
    def callback(xk):
         nonlocal num_iters
         num_iters+=1
    x, info = cg(A, b, x0=x0, tol=tol, callback=callback)
    # Devuelve la solución y el número de iteraciones
    return x, info, num_iters

def factorizacionLU(A, b):
    P, L, U = scipy.linalg.lu(A)
    # Paso 1: Resolver Ly = Pb
    y = np.linalg.solve(L, np.dot(P, b))
    # Paso 2: Resolver Ux = y
    x = np.linalg.solve(U, y)
    return x

def gauss_seidel(A, b,k=0, max_iter=1000000, xn=None, x0 = None, tol=1e-10):
    if xn is None:
        xn = np.zeros(len(A))
    if x0 is None:
        x0 = np.zeros(len(A))
    iteracion = 0
    for i in range(max_iter):
        x_ponderado = (k+1)*xn - k*x0
        iteracion = i
        x_prev = x_ponderado.copy()
        for j in range(len(x_ponderado)):
            x_ponderado[j] = (b[j] - np.dot(A[j, :j], x_ponderado[:j]) - np.dot(A[j, j+1:], x_prev[j+1:])) / A[j, j]
        if np.linalg.norm(x_ponderado - x_prev) < tol:
            return x_ponderado, iteracion 
        x0 = xn
        xn = x_ponderado
    return x_ponderado, iteracion

if __name__ == "__main__":
    k = 4
    p = 1
    #p = (np.sqrt(5)-1)/2

    A= np.loadtxt("3matriz.txt", dtype=float, delimiter=" ")
    b = np.loadtxt("3b.txt", dtype=float, delimiter=" ")

    x0 = np.zeros(len(A))

    start_time_gc = time.time()
    x_gc, _, num_iter_gc = conjugate_gradient(A, b, x0)
    end_time_gc = time.time()
    residuo_gc = np.linalg.norm(np.dot(A, x_gc) - b)

    start_time_flu = time.time()
    x_flu = factorizacionLU(A, b)
    end_time_flu = time.time()
    residuo_flu = np.linalg.norm(np.dot(A, x_flu) - b)

    start_time_gs = time.time()
    x_gs, iter_gs = gauss_seidel(A, b, x0)
    end_time_gs = time.time()
    residuo_gs = np.linalg.norm(np.dot(A, x_gs) - b)

    # Imprimir resultados
    print("Gradiente Conjugado - Solucion:", x_gc)
    print("Gradiente Conjugado - Tiempo:", end_time_gc - start_time_gc, "segundos")
    print("Gradiente Conjugado - Iteraciones:", num_iter_gc)
    print("Gradiente Conjugado - Residuo:", residuo_gc)

    print("Factorizacion LU - Solucion:", x_flu)
    print("Factorizacion LU - Tiempo:", end_time_flu - start_time_flu, "segundos")
    print("Factorizacion LU - Residuo:", residuo_flu)

    print("Gauss-Seidel  - Solucion:", x_gs)
    print("Gauss-Seidel - Tiempo:", end_time_gs - start_time_gs, "segundos")
    print("Gauss-Seidel - Iteraciones:", iter_gs)
    print("Gauss-Seidel - Residuo:", residuo_gs)