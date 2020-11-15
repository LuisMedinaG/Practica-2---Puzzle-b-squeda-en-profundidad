import numpy as np
import math
import cv2


class Vertex:
    def __init__(self, vert_id, x, y):
        self.vert_id = vert_id
        self.x = x
        self.y = y
        self.neighbors = []
        self.distance = -1
        self.previous = None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"ID: {self.vert_id} COORD: ({self.x},{self.y})"

    def __repr__(self):
        return f"ID: {self.vert_id} COORD: ({self.x},{self.y})"

    def addVertex(self, vertex):
        self.neighbors.append(vertex)

    def getNeighbors(self):
        return self.neighbors

    def distance(self, x2, y2):
        return math.sqrt((x2 - self.x1)**2 + (y2 - self.y1)**2)


class ShapeDetector:
    def __init__(self, img_path, img_length=700, img_height=500):
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, (img_length, img_height))
        self.shapes = []
        self.vertex_list = []

    def findShapes(self, lower_red, upper_red, lower_black, upper_black):
        color_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        mask_red = cv2.inRange(color_img, lower_red, upper_red)
        red_contours, h = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        self.shapes.extend(red_contours)

        mask_black = cv2.inRange(color_img, lower_black, upper_black)
        black_contours, h = cv2.findContours(mask_black, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        self.shapes.extend(black_contours)

    def findVertices(self):
        for shp in self.shapes:
            approx = cv2.approxPolyDP(shp, 0.009 * cv2.arcLength(shp, True),
                                      True)
            points = approx.ravel()

            shp_id = 0
            for i in range(0, len(points), 2):
                x = points[i]
                y = points[i + 1]

                nodo = Vertex(shp_id, x, y)
                self.vertex_list.append(nodo)
                shp_id += 1

    def findPossiblePaths(self):
        for source in self.vertex_list:
            center = (source.x, source.y + 5)
            cv2.putText(self.img, str(source.vert_id), center, cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 0, 255), 2)

            for target in self.vertex_list:
                if source != target and target not in source.neighbors:
                    if self.areConnectable(source, target):
                        source.addVertex(target)
                        target.addVertex(source)

    def areConnectable(self, vertex1, vertex2):
        pt_a = np.array([vertex1.x, vertex1.y])
        pt_b = np.array([vertex2.x, vertex2.y])
        dist = int(self.vertex1.distance(vertex2.x, vertex2.y))
        line = np.linspace(pt_a, pt_b, int(dist / 3), dtype="int")

        for i, point in enumerate(line):
            if i <= 2 or i > len(line) - 2:
                continue

            x = int(point[0])
            y = int(point[1])
            r, g, b = self.img[y][x]
            if self.isBlack(r, g, b):
                return False
        return True

    def drawLines(self, path):
        for idx, org_id in enumerate(path):
            if idx == len(path) - 1:
                break

            des_id = path[idx + 1]
            org = self.vertex_list[org_id]
            des = self.vertex_list[des_id]
            org_pos = (org.x, org.y)
            des_pos = (des.x, des.y)

            cv2.line(self.img, org_pos, des_pos, (0, 255, 0), thickness=1)
            cv2.circle(self.img, org_pos, 2, (0, 255, 255), thickness=2)

    def isBlack(self, r, g, b):
        return r < 150 and g < 150 and b < 150


def shortestPathBFS(vertex):
    """
    Shortest Path - Breadth First Search
    :param vertex: the starting graph node
    :return: does not return, changes in place
    """
    if vertex is None:
        return

    queue = []                  # our queue is a list with insert(0) as enqueue() and pop() as dequeue()
    queue.insert(0, vertex)

    while len(queue) > 0:
        current_vertex = queue.pop()                    # remove the next node in the queue
        next_distance = current_vertex.distance + 1     # the hypothetical distance of the neighboring node

        for neighbor in current_vertex.getNeighbors():
            if neighbor.distance == -1 or neighbor.distance > next_distance:    # try to minimize node distance
                neighbor.distance = next_distance       # distance is changed only if its shorter than the current
                neighbor.previous = current_vertex      # keep a record of previous vertexes so we can traverse our path
                queue.insert(0, neighbor)


def traverseShortestPath(target):
    """
    Traverses backward from target vertex to source vertex, storing all encountered vertex id's
    :param target: Vertex() Our target node
    :return: A list of all vertexes in the shortest path
    """
    vertexes_in_path = []

    while target.previous:
        vertexes_in_path.append(target.vert_id)
        target = target.previous

    return vertexes_in_path


def main():
    image_filename = 'aima_maze.png'
    sD = ShapeDetector(image_filename)

    # RGB value range of red and black objects
    lower_red = np.array([200, 20, 30])
    upper_red = np.array([240, 40, 50])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([20, 20, 20])

    sD.findShapes(lower_red, upper_red, lower_black, upper_black)
    sD.findVertices()
    sD.findPossiblePaths()

    # source = sD.vertex_list[0]
    # target = sD.vertex_list[1]

    # shortestPathBFS(source)
    # vertexes_in_path = traverseShortestPath(target)

    # # Display the results
    # print('shortest path length: ', len(vertexes_in_path))
    # print('shortest path: ', vertexes_in_path[::-1])

    cv2.imshow(image_filename, sD.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
