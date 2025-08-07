import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# función matemática declarada como anónima con lambda pasando th como argumento de un vector
func = lambda th: np.sin(1 / 2 * th[0] ** 2 - 1 / 4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1])

# generamos 100 valores entre -2 y 2
res = 100
_X = np.linspace(-2, 2, res)
_Y = np.linspace(-2, 2, res)

# creamos la matriz de zeros que guardará los resultados de los puntos x, y
_Z = np.zeros((res, res))

# iteramos sobre _X, _Y
for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        # guardamos el valor de ambas en _Z
        _Z[iy, ix] = func([x, y])

# pintamos el gráfico con los parámetros _X, _Y, _Z y lo ponemos bonito
plt.contourf(_X, _Y, _Z, 100)
plt.colorbar()

# generamos el punto aleatorio entre los valores 2 y -2
Theta = np.random.rand(2) * 4 -2

_T = np.copy(Theta)

# valor para incrementar
h = 0.001

# variable para guardar el valor de la derivada parcial calculada
grad = np.zeros(2)

# ratio de aprendizage (learning rate)
lr = 0.001

# punto inicial de color blanco
plt.plot(Theta[0], Theta[1], "o", c="white")

# iteraciones del descenso del gradiente
for _ in range(10000):
    # recoremos el vector Theta e incrementamos ligeramente con h
    for it, th in enumerate(Theta):
        # mostramos el vector original
        _T = np.copy(Theta)

        # le aplicamos el incremento con h
        _T[it] = _T[it] + h

        # comprobamos la diferencia entre la función original con la variante incrementada con h
        deriv = (func(_T) - func(Theta)) / h

        # guardamos la derivada parcial 
        grad[it] = deriv 

    # poco a poco se va estimando la derivada parcial
    Theta = Theta - lr * grad    
    
    # iteraciones del punto de color rojo
    if(_ % 100 == 0):
        plt.plot(Theta[0], Theta[1], ".", c="red")

# punto final de color verde
plt.plot(Theta[0], Theta[1], "o", c="green")
plt.show()
