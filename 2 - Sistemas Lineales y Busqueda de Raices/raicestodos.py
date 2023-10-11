import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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

# Función para encontrar las raíces de un polinomio dado por sus coeficientes
def find_roots(C1, C2, C3, C4):
    def equation(x):
        return C1 * x**3 + C2 * x**2 + C3 * x + C4

    # Encontrar las raíces utilizando fsolve
    roots = fsolve(equation, [-50, -10, 10, 50])  # Puedes ajustar los valores iniciales
    return roots

# Encontrar las raíces de todos los polinomios y almacenarlas en una lista
all_roots = [find_roots(C1, C2, C3, C4) for C1, C2, C3, C4 in zip(C1_values, C2_values, C3_values, C4_values)]

# Crear una lista de todas las raíces únicas
unique_roots = np.unique(np.concatenate(all_roots))

# Crear una lista de puntos de intersección entre los polinomios
intersection_points = []
for root in unique_roots:
    # Verificar si el punto está cerca de todos los polinomios
    if all(abs(np.polyval([C1, C2, C3, C4], root)) < 1e-3 for C1, C2, C3, C4 in zip(C1_values, C2_values, C3_values, C4_values)):
        intersection_points.append((root, 0))

# Graficar los puntos de intersección
for point in intersection_points:
    plt.plot(point[0], point[1], 'ro', label=f'Intersección en x={point[0]}')

# Graficar los polinomios
x = np.linspace(-50, 50, 1000)
for C1, C2, C3, C4 in zip(C1_values, C2_values, C3_values, C4_values):
    P_x = C1 * x**3 + C2 * x**2 + C3 * x + C4
    plt.plot(x, P_x, linewidth=2)

plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Intersecciones entre los Polinomios')
plt.axhline(0, color='black', linewidth=0.5)  # Eje y
plt.grid(True)

# Mostrar el gráfico
plt.legend()
plt.show()
