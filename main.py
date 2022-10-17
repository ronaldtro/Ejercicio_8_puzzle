import sys
import timeit

import numpy as np

from board import Board
from dfs import DFS
def readFile():
    datos = []
    lista = []
    with open("puzzle.txt") as fname:
        lineas = fname.readlines()
        for linea in lineas:
            datos.append(linea.strip('\n'))
    join = "".join(datos)
    for indice in range(len(join)):
        caracter = join[indice]
        if caracter != ",":
            lista.append(caracter)
    puzzle = ",".join(lista)

    return puzzle


def main():
    ini_time = timeit.default_timer()
    data = readFile()
    p = Board(np.array(eval(data)))
    s = DFS(p)
    s.solve()
    stop_time = timeit.default_timer()
    end_time = stop_time - ini_time
    consum =sys.getsizeof(DFS)
    consuF = sys.getsizeof(main)
    consuB = sys.getsizeof(Board)
    consumT = consum+consuF+consuB
    print('Camino realizado: ' + str(s.path) + '\n')
    print('Costo del camino: ' + str(len(s.path)) + '\n')
    print('Nodos expandidos: ' + str(s.nodes_expanded) + '\n')
    print('Nodos Explorados: ' + str(len(s.explored_nodes)) + '\n')
    print('Profundida de busqueda:' + str(s.solution.depth) + '\n')
    print('Maxima profundidad de busqueda: ' + str(s.max_depth) + '\n')
    print("Tiempo de ejecucion : ", format(end_time, '.8f'))
    print('Memoria Utilizada',consumT,"Bytes")



if __name__ == "__main__":
    main()
