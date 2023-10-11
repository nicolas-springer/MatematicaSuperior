import numpy as np
import matplotlib.pyplot as plt
import math 
import sympy as sp


with open('EJE2.txt', 'r') as file:
    lines = file.readlines()

C1_values = []
C2_values = []
C3_values = []
C4_values = []

polinomios = []

for line in lines:
    values = line.split()
    C1 = float(values[0])
    C2 = float(values[1])
    C3 = float(values[2])
    C4 = float(values[3])
    
    P_x = np.poly1d([C1, C2, C3, C4])
    polinomios.append(P_x)


x = np.linspace(-50, 50, 1000) 

# Graficar P(x) 


for i, polinomio in enumerate(polinomios):
    plt.plot(x, polinomio(x), label=f'P{i+1}(x) = {polinomio}')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Gráfico de P(x) para Diferentes Coeficientes')
plt.legend()
plt.xlim(-20, 20)  # Cambia los límites del eje x
plt.ylim(-20, 20)  # Cambia los límites del eje y
plt.axhline(0, color='black', linewidth=0.5)  # Eje y
plt.axvline(0, color='black', linewidth=0.5)  # Eje x
plt.grid(True)

plt.show()


#############################################

# Método de Newton-Raphson

def metodoNewtonRaphson(funcion, funcionDerivada, x0, tolerancia, cantMaxIteraciones):
    x = x0
    aproximaciones = [] 

    for i in range(cantMaxIteraciones):
        fdeX = funcion(x)
        fPrimaDeX = funcionDerivada(x)
        
        aproximaciones.append(x)

        if abs(fdeX) < tolerancia:
            return x, aproximaciones, i  
        x = x - fdeX / fPrimaDeX
   
    raise ValueError("El método de Newton-Raphson no convergió después de {} iteraciones.".format(cantMaxIteraciones))


def metodoBiseccion(funcion, a, b, cantMaxIteraciones, tolerancia):
    i = 0
    err = []
    resultados = []

    m = (a + b) / 2.0
    q = funcion(m)
    error = abs((m - a) / 2)
    err.append(error)
    resultados.append(m)

    while i < cantMaxIteraciones and abs(err[-1]) > tolerancia:
        m = (a + b) / 2.0
        q = funcion(m)

        if np.sign(funcion(a)) == np.sign(q):
            a = m
        else:
            b = m

        error = abs((a - b) / 2)
        err.append(error)
        resultados.append(m)
        i += 1

    return resultados,  i


def combinacionDeMetodos(func, func_prime, a, b, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton):

    # Ejecutar método de bisección 
    resultados_biseccion, iteraciones_biseccion = metodoBiseccion(func, a, b, max_iter_biseccion, tolerancia_biseccion)

    # Tomar la última aproximación de la bisección como valor inicial para Newton-Raphson
    x0_newton = resultados_biseccion[-1]

    # Ejecutar método de Newton-Raphson con x0 obtenido de la bisección
    raiz, aproximaciones_newton, iteraciones_newton = metodoNewtonRaphson(func, func_prime, x0_newton, tolerancia_newton, max_iter_newton)

    total_iteraciones = iteraciones_biseccion + iteraciones_newton

    return raiz


######################################################################


### PUNTO 1 ###

# Valores iniciales
a_biseccion = -1
b_biseccion = 20
max_iter_biseccion = 100
tolerancia_biseccion = 1
total_iteraciones = 0
tolerancia_newton = 1e-8 
max_iter_newton = 100

funcionDerivada = np.polyder(polinomios[7]-polinomios[6])

punto1 = combinacionDeMetodos(polinomios[7] - polinomios[6], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 2 ###

# Valores iniciales
a_biseccion = 1
b_biseccion = 3

funcionDerivada = np.polyder(polinomios[1]-polinomios[7])

punto2 = combinacionDeMetodos(polinomios[1] - polinomios[7], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 3 ###

# Valores iniciales
a_biseccion = 1
b_biseccion = 2


funcionDerivada = np.polyder(polinomios[1]-polinomios[8])

punto3 = combinacionDeMetodos(polinomios[1] - polinomios[8], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)



### PUNTO 4 ###

# Valores iniciales
a_biseccion = 1
b_biseccion = 2


funcionDerivada = np.polyder(polinomios[4]-polinomios[8])

punto4 = combinacionDeMetodos(polinomios[4] - polinomios[8], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 5 ###

# Valores iniciales
a_biseccion = -30
b_biseccion = -17
tolerancia_newton = 0.1

funcionDerivada = np.polyder(polinomios[4]-polinomios[0])

punto5 = combinacionDeMetodos(polinomios[4] - polinomios[0], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 6 ###

# Valores iniciales
a_biseccion = -5
b_biseccion = 2
tolerancia_newton = 1e-8

funcionDerivada = np.polyder(polinomios[3]-polinomios[0])

punto6 = combinacionDeMetodos(polinomios[3] - polinomios[0], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 7 ###

# Valores iniciales
a_biseccion = -5
b_biseccion = 1

funcionDerivada = np.polyder(polinomios[3]-polinomios[5])

punto7 = combinacionDeMetodos(polinomios[3] - polinomios[5], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 8 ###

# Valores iniciales
a_biseccion = 6
b_biseccion = 8

funcionDerivada = np.polyder(polinomios[2]-polinomios[5])

punto8 = combinacionDeMetodos(polinomios[2] - polinomios[5], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


### PUNTO 9 ###

# Valores iniciales
a_biseccion = 2
b_biseccion = 4

funcionDerivada = np.polyder(polinomios[2]-polinomios[6])

punto9= combinacionDeMetodos(polinomios[2] - polinomios[6], funcionDerivada, a_biseccion, b_biseccion, max_iter_biseccion, tolerancia_biseccion, max_iter_newton, tolerancia_newton)


# Crear un rango de valores de x 
x_intervalo_1a2 = np.linspace(punto1, punto2, 1000)
x_intervalo2a3 = np.linspace(punto2, punto3, 1000)
x_intervalo3a4 = np.linspace(punto3, punto4, 1000)
x_intervalo4a5 = np.linspace(punto4, punto5, 1000)
x_intervalo5a6 = np.linspace(punto5, punto6, 1000)
x_intervalo6a7 = np.linspace(punto6, punto7, 1000)
x_intervalo7a8 = np.linspace(punto7, punto8, 1000)
x_intervalo8a9= np.linspace(punto8, punto9, 1000)
x_intervalo9a1= np.linspace(punto9, punto1, 1000)

polinomio0 = polinomios[0]
polinomio1 = polinomios[1]
polinomio2 = polinomios[2]
polinomio3 = polinomios[3]
polinomio4 = polinomios[4]
polinomio5 = polinomios[5]
polinomio6 = polinomios[6]
polinomio7 = polinomios[7]
polinomio8 = polinomios[8]

# Calcular los valores de P(x) correspondientes

P_x7 = polinomio7(x_intervalo_1a2)
P_x1 = polinomio1(x_intervalo2a3)
P_x8 = polinomio8(x_intervalo3a4)
P_x4 = polinomio4(x_intervalo4a5)
P_x0 = polinomio0(x_intervalo5a6)
P_x3 = polinomio3(x_intervalo6a7)
P_x5 = polinomio5(x_intervalo7a8)
P_x2 = polinomio2(x_intervalo8a9)
P_x6 = polinomio6(x_intervalo9a1)


# Graficar del colibri resultante de las intersecciones
plt.plot(x_intervalo_1a2, P_x7, color='blue')
plt.plot(x_intervalo2a3, P_x1, color='blue')
plt.plot(x_intervalo3a4, P_x8, color='blue')
plt.plot(x_intervalo4a5, P_x4, color='blue')
plt.plot(x_intervalo5a6, P_x0, color='blue')
plt.plot(x_intervalo6a7, P_x3, color='blue')
plt.plot(x_intervalo7a8, P_x5, color='blue')
plt.plot(x_intervalo8a9, P_x2, color='blue')
plt.plot(x_intervalo9a1, P_x6, color='blue')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Colibri')
plt.xlim(-20, 20)  # Cambia los límites del eje x
plt.ylim(-20, 20)  # Cambia los límites del eje y
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)

plt.show()


