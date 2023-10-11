import numpy as np

def generar_matriz_diagonalmente_dominante(N, min_valor, max_valor):
    # Genera una matriz aleatoria de NxN con números enteros en el rango [min_valor, max_valor]
    matriz = np.random.randint(min_valor, max_valor + 1, size=(N, N))
    # Asegura que la matriz sea diagonalmente dominante
    for i in range(N):
        suma_fila = np.sum(np.abs(matriz[i])) - np.abs(matriz[i, i])  # Suma de los elementos fuera de la diagonal
        if np.abs(matriz[i, i]) <= suma_fila:
            # Ajusta el elemento de la diagonal principal para que sea mayor que la suma de los otros elementos
            matriz[i, i] = suma_fila + 1
    return matriz

N = 6
min_valor = 1
max_valor = N
matriz_generada = generar_matriz_diagonalmente_dominante(N, min_valor, max_valor)
b = np.random.rand(N)
# También puedes guardar la matriz en un archivo de texto si lo deseas
np.savetxt("2matriz.txt", matriz_generada, fmt='%.8f', delimiter=' ')
np.savetxt("2b.txt", b, fmt='%.8f', delimiter=' ')
