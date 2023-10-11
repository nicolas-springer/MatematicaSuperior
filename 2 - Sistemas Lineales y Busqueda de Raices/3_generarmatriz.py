import numpy as np

def generar_matriz_H_(p, k):
    n = k + 1
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                elemento = 1
            else:
                elemento = 1 / (p + i + j)
            A[i][j] = elemento
    b = np.random.rand(n)
    return A, b

p_valor = 1
#p_valor = (np.sqrt(5)-1)/2
k_valor = 4

A_generada, b_generado = generar_matriz_H_(p_valor, k_valor)
print(A_generada)
print(b_generado)
# Guardar la matriz en un archivo de texto
np.savetxt("3matriz.txt", A_generada, fmt='%.8f', delimiter=' ')
np.savetxt("3b.txt", b_generado, fmt='%.8f', delimiter=' ')