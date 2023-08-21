import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos desde Colibri.csv
with open('Colibri.csv', 'rb') as file:
    data = np.genfromtxt(file, delimiter=';', skip_header=2, usecols=(0, 1, 2), converters={1: lambda s: float(s.decode('utf-8').replace(',', '.')), 2: lambda s: float(s.decode('utf-8').replace(',', '.'))})

t = data[:, 0]  # Valores de t
Z = data[:, 1] + 1j*data[:, 2]  # Valores complejos de Z

# Paso 2: Graficar los valores complejos de Z en el plano complejo
plt.figure("Ejercicio 1")
plt.scatter(Z.real, Z.imag)
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.title('Representación en el Plano Complejo')
plt.grid(True)

# Paso 3: Aproximar la función original usando la fórmula de la suma de Riemann
M = 68
omega_0 = 2*np.pi/(t[-1]-t[0])  # Frecuencia fundamental
zeta = np.zeros((2*M+1,), dtype=np.complex128)  # Coeficientes de Fourier
h = (t[-1] - t[0]) / len(t) # (b-a) / n

for k in range(-M, M+1): 
    Z_k = Z[1:]*np.exp(-1j*k*omega_0*t[1:])
    zeta[k+M] = h * np.sum((Z_k[1:] + Z_k[:-1])/2) 

# Ejemplo: 
# Z_k[1:]: [1.0, 2.0, 3.0, 4.0] # Todos los valores de Z_k excepto el último
# Z_k[:-1]:  [2.0, 3.0, 4.0, 5.0] # Todos los valores de Z_k excepto el primero
# Z_k[1:] + Z_k[:-1] = [3.0, 5.0, 7.0, 9.0]
# (Z_k[:-1] + Z_k[1:]) / 2: =  [1.5, 2.5, 3.5, 4.5]

# Paso 4: Construir la aproximación y graficarla junto con los datos originales
t_approx = np.linspace(t[0], t[-1], 1000)
C_approx = np.zeros_like(t_approx, dtype=np.complex128)
for k in range(-M, M+1):
    C_approx += zeta[k+M]*np.exp(1j*k*omega_0*t_approx)

# Escalar la aproximación para que coincida con los datos originales
factor_escala = np.abs(Z).max() / np.abs(C_approx).max()
C_approx *= factor_escala

plt.figure("Ejercicio 2")
plt.plot(t_approx, C_approx.real, color="red", label='Aproximación')
plt.scatter(t, Z.real, label='Datos Originales')
plt.vlines(t, ymin=0, ymax=Z.real, colors=["grey"], linestyle='--', alpha=0.5)
plt.xlabel('t')
plt.ylabel('Parte Real')
plt.title('Aproximación de la Función Original')
plt.legend()
plt.grid(True)

# Paso 5: Mostrar la aproximación formando el colibri 
plt.figure()
plt.plot(C_approx.real, C_approx.imag, color="red", label='Aproximación')
plt.scatter(Z.real, Z.imag, label='Datos Originales')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.title('Aproximación formando el Colibri')
plt.legend()
plt.grid(True)

plt.show()
