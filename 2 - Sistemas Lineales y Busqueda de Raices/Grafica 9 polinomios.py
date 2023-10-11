import numpy as np
import matplotlib.pyplot as plt

# Leer los datos del archivo
with open('EJE2.txt', 'r') as file:
    lines = file.readlines()

# Inicializar listas para almacenar los coeficientes C1, C2, C3 y C4
C1_values = []
C2_values = []
C3_values = []
C4_values = []

# Procesar las líneas del archivo y extraer los valores
for line in lines:
    values = line.split()
    C1_values.append(float(values[0]))
    C2_values.append(float(values[1]))
    C3_values.append(float(values[2]))
    C4_values.append(float(values[3]))

# Crear un rango de valores de x
x = np.linspace(-50, 50, 1000)  # Puedes ajustar el rango según tus necesidades

# Graficar P(x) para cada conjunto de coeficientes
for C1, C2, C3, C4 in zip(C1_values, C2_values, C3_values, C4_values):
    P_x = C1 * x**3 + C2 * x**2 + C3 * x + C4
    plt.plot(x, P_x, label=f'P(x) = {C1}x^3 + {C2}x^2 + {C3}x + {C4}')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Gráfico de P(x) para Diferentes Coeficientes')
plt.legend()
plt.xlim(-15, 15)  # Cambia los límites del eje x
plt.ylim(-15, 15)  # Cambia los límites del eje y
plt.axhline(0, color='black', linewidth=0.5)  # Eje y
plt.axvline(0, color='black', linewidth=0.5)  # Eje x
plt.grid(True)

# Mostrar el gráfico
plt.show()