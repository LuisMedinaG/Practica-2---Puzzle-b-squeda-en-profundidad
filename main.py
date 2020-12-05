import numpy as np
import heapq
import math
import cv2


class Vertex:
    def __init__(self, vert_id, x, y):
        self.vert_id = vert_id
        self.x = x
        self.y = y

        self.neighbors = []
        self.distance = -1
        self.curr_dist = math.inf
        self.previous = None

        self.gscore = math.inf
        self.fscore = math.inf
        self.closed = False
        self.out_openset = True

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"ID: {self.vert_id} COORD: ({self.x},{self.y})"

    def __repr__(self):
        return f"ID: {self.vert_id} COORD: ({self.x},{self.y})"

    def __lt__(self, b):
        return self.fscore < b.fscore

    def addVertex(self, vertex):
        self.neighbors.append(vertex)

    def getDistance(self, other_vertex):
        return math.sqrt((other_vertex.x - self.x)**2 +
                         (other_vertex.y - self.y)**2)

    def getPos(self):
        return (self.x, self.y)


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
        shp_id = 0
        for shp_idx, shp in enumerate(self.shapes):
            approx = cv2.approxPolyDP(shp, 0.009 * cv2.arcLength(shp, True),
                                      True)
            points = approx.ravel()
            for pt_idx in range(0, len(points), 2):
                x = points[pt_idx]
                y = points[pt_idx + 1]

                nodo = Vertex(shp_id, x, y)
                self.vertex_list.append(nodo)
                shp_id += 1

                if shp_idx < 2:
                    break

    def findPossiblePaths(self):
        for source in self.vertex_list:
            for target in self.vertex_list:
                if source != target and target not in source.neighbors:
                    if self.areConnectable(source, target):
                        source.addVertex(target)
                        target.addVertex(source)

    def areConnectable(self, vertex1, vertex2, r=10):
        pt_a = np.array([vertex1.x, vertex1.y])
        pt_b = np.array([vertex2.x, vertex2.y])
        dist = int(vertex1.getDistance(vertex2))
        line = np.linspace(pt_a, pt_b, int(dist / 2), dtype="int")

        for i, point in enumerate(line):
            x = int(point[0])
            y = int(point[1])
            r, g, b = self.img[y][x]

            # After certain radius start checking pixel values
            if i <= 3 or i >= len(line) - 3:
                continue

            # Other way of checking the skip radius
            # if vertex1.getDistance(x, y) <= r or vertex2.getDistance(x, y) <= r:
            #     continue

            if self.isBlack(r, g, b):
                return False
        return True

    def drawLines(self, path, drawDist=False, drawId=False, drawVertex=False):
        for i, vert_id in enumerate(path):
            org = self.vertex_list[vert_id]

            if drawId:
                cv2.putText(self.img, str(org.vert_id), org.getPos(),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 169), 2)
            if drawVertex:
                cv2.circle(
                    self.img, org.getPos(), 2, (0, 255, 255), thickness=2)
            if i == len(path) - 1:
                break

            des = self.vertex_list[path[i + 1]]

            midX = int((org.x + des.x) / 2)
            midY = int((org.y + des.y) / 2)
            dist = str(int(org.getDistance(des)))

            cv2.line(
                self.img, org.getPos(), des.getPos(), (0, 255, 0), thickness=1)

            if drawDist:
                cv2.putText(self.img, dist, (midX, midY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (184, 84, 39), 2)

    def isBlack(self, r, g, b):
        return r < 150 and g < 150 and b < 150


def shortestPathBFS(start):
    """
    Shortest Path - Breadth First Search
    """
    if start is None:
        return None

    # keep track of nodes to be checked
    queue = [start]
    start.curr_dist = 0

    while queue:
        curr = queue.pop()
        for neighbor in curr.neighbors:
            next_distance = curr.curr_dist + curr.getDistance(neighbor)
            if neighbor.curr_dist == math.inf or neighbor.curr_dist > next_distance:
                neighbor.curr_dist = next_distance
                neighbor.previous = curr
                queue.insert(0, neighbor)


def heuristic(cell, goal):
    """computes the 'direct' distance between two (x,y) tuples"""
    return math.hypot(goal.x - cell.x, goal.y - cell.y)


def AStar(start, goal):
    if start == goal:
        return

    start.fscore = heuristic(start, goal)
    start.gscore = 0

    open_set = []
    heapq.heappush(open_set, start)

    while open_set:
        current = heapq.heappop(open_set)
        if current == goal:
            return

        current.out_openset = True
        current.closed = True

        for neighbor in current.neighbors:
            if neighbor.closed:
                continue
            
            tentative_gscore = current.gscore + current.getDistance(neighbor)
            if tentative_gscore >= neighbor.gscore:
                continue

            neighbor.previous = current
            neighbor.gscore = tentative_gscore
            neighbor.fscore = tentative_gscore + heuristic(neighbor, goal)

            if neighbor.out_openset:
                neighbor.out_openset = False
                heapq.heappush(open_set, neighbor)
            else:
                open_set.remove(neighbor)
                heapq.heappush(open_set, neighbor)

    print("No se encontro camino")

def bestFirst(start):
    if start is None:
        return None
        
    # keep track of nodes to be checked
    queue = [start]
    start.curr_dist = 0
    while queue:
        curr = queue.pop()
        for neighbor in curr.neighbors:
            next_distance = curr.curr_dist + curr.getDistance(neighbor)
            if neighbor.curr_dist == math.inf or neighbor.curr_dist > next_distance:
                neighbor.curr_dist = next_distance
                neighbor.previous = curr
                queue.insert(0, neighbor)
        queue = (sorted(queue, key=lambda x: x.curr_dist, reverse=True))

def traverseShortestPath(target):
    vertexes_in_path = []

    while target.previous:
        vertexes_in_path.append(target.vert_id)
        target = target.previous

    return vertexes_in_path


def main():
    image_filename = 'aima_maze_modified.png'
    sD = ShapeDetector(image_filename)

    # RGB value range of red and black objects
    lower_red = np.array([100, 50, 50])
    upper_red = np.array([255, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([20, 20, 20])

    sD.findShapes(lower_red, upper_red, lower_black, upper_black)
    sD.findVertices()
    sD.findPossiblePaths()

    # The first and second element are source and target
    source = sD.vertex_list[0]
    target = sD.vertex_list[1]

    print(source)
    print(target)
    # print(sD.vertex_list)
    
    # bestFirst(source)
    AStar(source, target)
    vertexes_in_path = traverseShortestPath(target)
    vertexes_in_path.append(0)
    sD.drawLines(vertexes_in_path, True, True)

    # Display the results
    print('shortest path length: ', len(vertexes_in_path))
    print('shortest path vertexs IDs: ', vertexes_in_path[::-1])

    # Save the image
    cv2.imwrite('bfs.png', sD.img)
    cv2.imshow(image_filename, sD.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
