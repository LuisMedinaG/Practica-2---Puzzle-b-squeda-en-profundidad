import numpy as np
from collections import defaultdict
import math
import cv2
import copy


class vertex:
    def __init__(self, id, x, y):
        self.id = str(id)
        self.x = x
        self.y = y
        self.lista_adyancecia = []

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"{self.id} ({self.x},{self.y})"

    def __repr__(self):
        return f"({self.x},{self.y})"


class Graph:
    def __init__(self):
        #self.V = vertices
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)


def findShapes(img):
    color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([50, 50, 50])
    upper_red = np.array([255, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])

    mask_red = cv2.inRange(color, lower_red, upper_red)
    red_contours, h = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    mask_black = cv2.inRange(color, lower_black, upper_black)
    black_contours, h = cv2.findContours(mask_black, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

    return red_contours, black_contours


def makeGraph(red_contours, black_contours):
    vertex_list = []
    
    if red_contours:
        org = red_contours[0]
        approx = cv2.approxPolyDP(org, 0.009 * cv2.arcLength(org, True), True)
        n = approx.ravel()
        x = n[-6]
        y = n[-5]
        nodo = vertex(len(vertex_list), x, y)
        vertex_list.append(nodo)

        des = red_contours[1]
        approx = cv2.approxPolyDP(des, 0.009 * cv2.arcLength(des, True), True)
        n = approx.ravel()
        x = n[-2]
        y = n[-1]
        nodo = vertex(len(vertex_list), x, y)
        vertex_list.append(nodo)

    for cnt in black_contours:
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)

        n = approx.ravel()
        i = 0
        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                nodo = vertex(len(vertex_list), x, y)
                vertex_list.append(nodo)
            i = i + 1

    return vertex_list


def findObstacles(vertex_list, img):
    for origen in vertex_list:
        cv2.putText(img, origen.id, (origen.x, origen.y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,0), 1)

        for destino in vertex_list:
            if origen != destino and destino not in origen.lista_adyancecia:
                if areConectable(origen, destino, img):
                    origen.lista_adyancecia.append(destino)
                    destino.lista_adyancecia.append(origen)


def hypot(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def areConectable(origen, destino, img):
    pt_a = np.array([origen.x, origen.y])
    pt_b = np.array([destino.x, destino.y])
    dist = int(hypot(origen.x, origen.y, destino.x, destino.y))
    line = np.linspace(pt_a, pt_b, int(dist / 3), dtype="int")

    for i, point in enumerate(line):
        if i <= 3 or i > len(line) - 3:
            continue
        x = int(point[0])
        y = int(point[1])

        r, g, b = img[y][x]
        if isBlack(r, g, b):
            return False
    return True


def isBlack(r, g, b):
    return r < 50 and g < 50 and b < 50


def dfs(root, goal):
    stack = [(root, [root.id])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex.id not in visited:
            if vertex == goal:
                return path
            visited.add(vertex.id)
            for neighbor in reversed(vertex.lista_adyancecia):
                stack.append((neighbor, path + [neighbor.id]))
    return []


def toDict(vertex_list):
    #g = Graph()
    graph = {}
    for vertex in vertex_list:
        ver = vertex.id
        #print(vertex.id)
        for adyacente in vertex.lista_adyancecia:
            print("ad", adyacente)
            ady = adyacente
            #graph[] = str(ady)
            #print("Vertex ", ver)
            #print("Ady ",ady)
            #g.addEdge(ver,ady)
    #return g.graph
    print(graph)


def DLS(lista, root, target, maxDepth):
    path = set()
    if root == target: return True

    if maxDepth <= 0: return False

    for i in lista[root]:
        path.add(i)
        #print('i =  %d / target = %d / maxDepth = %d' % (i, target, maxDepth - 1))
        if (DLS(lista, i, target, maxDepth - 1)):
            return True
    #print(path)
    return False


def IDDFS(lista, root, target, maxDepth):
    for i in range(maxDepth):
        if (DLS(lista, root, target, i)):
            return True
    return False


def drawLines(vertex_list, id_list, img):
    for i, vertex in enumerate(id_list):
        if i == len(id_list) - 1:
            break
        j = int(vertex)
        k = int(id_list[i+1])
        
        curr = vertex_list[j]
        next_v = vertex_list[k]

        cv2.line(
            img, (curr.x, curr.y), (next_v.x, next_v.y), (0, 255, 0),
            thickness=2)
        # break


def main():
    img_path = 'aima_maze.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (700, 500))

    red_contours, black_contours = findShapes(img)
    vertex_list = makeGraph(red_contours, black_contours)

    if vertex_list and red_contours:
        findObstacles(vertex_list, img)

        # ORIGEN Y DESTINO DEL MAPA
        root = vertex_list[0]
        target = vertex_list[1]

        # Implenetacion del algoritmo DFS
        path = dfs(root, target)
        print(path)
        drawLines(vertex_list, path, img)

        # Implenetacion del algoritmo IDDFS
        # ad_list = toDict(vertex_list)
        # print(ad_list)
        # ya nomas queria imprimir la imagen con los ids :)
        # IDDFS(ad_list, root, target, 1000)

        cv2.imshow(img_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
