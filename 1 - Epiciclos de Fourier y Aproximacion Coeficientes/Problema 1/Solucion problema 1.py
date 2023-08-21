import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re

# Convierte una cadena con formato "real + imag" a un número complejo.
def complex_from_str(s):
    real, imag = re.findall(r'[-+]?\d*\.\d+|\d+', s)
    return complex(float(real), float(imag))

# Traza la curva cerrada y los epiciclos basados en los radios y los coeficientes complejos Xk (Ck).
def plot_epicycles(radios, Xk, num_points=5000, T=10, dt=0.001):
    N = len(radios)
    t_values = np.linspace(0, 2 * np.pi, num_points)
    z_t = np.zeros(num_points, dtype=complex)

    # Calculamos las coordenadas complejas z(t) usando los radios y coeficientes complejos
    for i, t in enumerate(t_values):
        z_t[i] = np.sum([Xk[k] * np.exp(1j * k * t) for k in range(-N // 2, N // 2)])

    x_values = np.real(z_t)
    y_values = np.imag(z_t)
    print(len(x_values))
    # Creamos la figura una sola vez antes de la animación
    fig, ax = plt.subplots(figsize=(8, 8))
    line_fixed, = ax.plot(x_values, y_values, '-b', label='Figura')
    line_animation, = ax.plot([], [], '-r', label='Animación')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    # Función para inicializar la animación
    def init():
        line_animation.set_data([], [])
        return line_animation,

    # Función para animar los epiciclos
    def animate(i):
        line_animation.set_data(np.real(z_t[:i]), np.imag(z_t[:i]))
        return line_animation,

    # Configuramos los tiempos máximos y el paso de tiempo para la animación
    T_max = T
    dt_anim = dt

    # Calculamos el número de pasos para la animación
    num_steps = int(T_max / dt_anim)

    # Creamos la animación
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=dt_anim*100, blit=True)

    plt.show()

# Lee los datos desde el archivo epiciclos.txt el cual formateamos para que sea radio + "espacio en blanco" + (parte real+parte compleja j)
file_path = 'epiciclos.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

radios = []
Xk = []

# Extraer los radios y coeficientes complejos de cada línea del archivo
for line in lines[1:]:
    line_data = line.strip().split()
    radios.append(float(line_data[0]))
    Xk.append(complex_from_str(line_data[1]))

# Utilizamos la función plot_epicycles con los datos del archivo epiciclos.txt para obtener y graficar las coordenadas x e y.
plot_epicycles(radios, Xk)