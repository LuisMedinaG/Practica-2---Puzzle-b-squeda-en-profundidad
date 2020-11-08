import numpy as np
import math
import cv2
import copy


class vertex:
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_
        self.lista_adyancecia = []

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"({self.x},{self.y})"

    def __repr__(self):
        return f"({self.x},{self.y})"


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
        nodo = vertex(x, y)
        vertex_list.append(nodo)

        des = red_contours[1]
        approx = cv2.approxPolyDP(des, 0.009 * cv2.arcLength(des, True), True)
        n = approx.ravel()
        x = n[-2]
        y = n[-1]
        nodo = vertex(x, y)
        vertex_list.append(nodo)

    for cnt in black_contours:
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)

        n = approx.ravel()
        i = 0
        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                nodo = vertex(x, y)
                vertex_list.append(nodo)
            i = i + 1

    return vertex_list


def findObstacles(vertex_list, img):
    for origen in vertex_list:
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
    visited = set()
    stack = list(root)
    while stack:
        node = stack.pop()
        visited.add(node)

        if node == goal:
            return stack

        neighbours = reversed(node.lista_adyancecia)
        for curr in neighbours:
            if curr not in visited:
                stack.append(curr)
                visited.add(curr)


def DLS(self, root, target, maxDepth):

    if root == target: return True

    if maxDepth <= 0: return False

    for i in self.lista_adyancecia[root]:
        print('i =  %d / target = %d / maxDepth = %d' % (i, target,
                                                         maxDepth - 1))
        if (self.DLS(i, target, maxDepth - 1)):
            return True
    return False


def IDDFS(self, root, target, maxDepth):

    for i in range(maxDepth):
        if (self.DLS(root, target, i)):
            return True
    return False


def printLines(vertex_list, img):
    for v_ori in vertex_list:
        for v_des in v_ori.lista_adyancecia:
            cv2.line(
                img, (v_ori.x, v_ori.y), (v_des.x, v_des.y), (0, 255, 0),
                thickness=1)


def main():
    path = 'aima_maze.png'
    img = cv2.imread(path)
    img = cv2.resize(img, (700, 500))

    red_contours, black_contours = findShapes(img)
    vertex_list = makeGraph(red_contours, black_contours)

    if vertex_list:
        findObstacles(vertex_list, img)
        printLines(vertex_list, img)

        cv2.imshow(path,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
