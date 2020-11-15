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
    def __init__(self, vertices):
        #self.V = vertices
        self.graph = defaultdict(list)

        for v in vertices:
            self.graph[v.id] = [[], set()]

    def addEdge(self, u, v):
        self.graph[u][0].append(v)


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
        center = (origen.x, origen.y)
        cv2.putText(img, origen.id, center, cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (255, 0, 0), 1)
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
        if i <= 2 or i > len(line) - 2:
            continue
        x = int(point[0])
        y = int(point[1])

        r, g, b = img[y][x]
        if isBlack(r, g, b):
            return False
    return True


def isBlack(r, g, b):
    return r < 150 and g < 150 and b < 150


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
    g = Graph(vertex_list)
    for vertex in vertex_list:
        ver = vertex.id
        for adyacente in vertex.lista_adyancecia:
            ady = adyacente.id
            g.addEdge(ver, ady)
    return g.graph


def DLS(lista, root, target, maxDepth):
    if root == target:
        return True
    if maxDepth <= 0:
        return False

    for i in lista[root][0]:
        lista[root][1].add(i)
        if (DLS(lista, i, target, maxDepth - 1)):
            print(lista[root][1])
            return True
    return False


def IDDFS(lista, root, target, maxDepth):
    for i in range(maxDepth):
        if (DLS(lista, root, target, i)):
            return True
    return False


def drawLines(vertex_list, path, img):
    for i, id in enumerate(path):
        if i == len(path) - 1:
            break
        j = int(id)
        k = int(path[i + 1])

        curr = vertex_list[j]
        next_v = vertex_list[k]

        center = (curr.x, curr.y)
        green = (0, 255, 0)
        cv2.line(img, center, (next_v.x, next_v.y), green, thickness=1)
        cv2.circle(img, center, 2, (0, 255, 255), thickness=2)


def main():
    img_path = 'aima_maze.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (700, 500))
    img2 = copy.deepcopy(img)

    red_contours, black_contours = findShapes(img)
    vertex_list = makeGraph(red_contours, black_contours)

    if vertex_list and red_contours:
        findObstacles(vertex_list, img)

        # ORIGEN Y DESTINO DEL MAPA
        root = vertex_list[0]
        target = vertex_list[1]

        # Implenetacion del algoritmo DFS
        path = dfs(root, target)
        print("PATH DE DFS: ")
        print(path)
        drawLines(vertex_list, path, img)
        cv2.imwrite('dfs.png', img)

        # Implenetacion del algoritmo IDDFS
        ad_list = toDict(vertex_list)
        print("PATH DE DFS ITERATIVA: ")
        IDDFS(ad_list, root.id, target.id, 200)
        drawLines(vertex_list, path, img2)
        cv2.imwrite('iddfs.png', img2)


if __name__ == '__main__':
    main()
