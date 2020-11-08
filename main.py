import numpy as np
import cv2


class vertex:
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_
        self.lista_adyancecia = []


def findShpaes(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.CHAIN_APPROX_NONE)
    contours, h = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
    return img, contours


def drawAndShowImg(img, contours):
    # Going through every contours found in the image.
    # for cnt in contours:
    #     # draws boundary of contours.
    #     cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def makeGraph(contours, img):
    vertex_list = []
    
    for cnt in contours:
        # Hacer el polinomio aproximado
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
        
        # Por cada vertice de la figura
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


def main():
    path = 'aima_maze.png'
    img, contours = findShpaes(path)

    vertex_list = []
    vertex_list = makeGraph(contours, img)
    print(vertex_list)

    drawAndShowImg(img, contours)


if __name__ == '__main__':
    main()
