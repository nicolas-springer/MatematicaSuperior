import numpy as np

def distancia_entre_puntos(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def puntos_mas_cercano_y_mas_lejano(lista_puntos):
    if len(lista_puntos) < 2:
        return None, None

    par_mas_cercano = None
    distancia_mas_corta = float('inf')
    par_mas_lejano = None
    distancia_mas_larga = 0

    for i in range(len(lista_puntos)):
        for j in range(i+1, len(lista_puntos)):
            distancia = distancia_entre_puntos(lista_puntos[i], lista_puntos[j])

            if distancia < distancia_mas_corta:
                distancia_mas_corta = distancia
                par_mas_cercano = (lista_puntos[i], lista_puntos[j])

            if distancia > distancia_mas_larga:
                distancia_mas_larga = distancia
                par_mas_lejano = (lista_puntos[i], lista_puntos[j])

    return par_mas_cercano, par_mas_lejano

puntos = [(14.332036703739252  ,  12.93970443215838),
        (4, 6),
        (7, 8), 
        (3, 5),
        (10, 12)]
par_cercano, par_lejano = puntos_mas_cercano_y_mas_lejano(puntos)

print("Par más cercano:", par_cercano)
print("Par más lejano:", par_lejano)
