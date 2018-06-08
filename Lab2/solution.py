import numpy as np
import math
import draw

class Mesh():
    def __init__(self, faces, coordinates = None):
        self.faces = faces
        vertices = set(i for f in faces for i in f)
        self.n = max(vertices)+1
        if coordinates != None:
            self.coordinates = np.array(coordinates)

        assert set(range(self.n)) == vertices
        for f in faces:
            assert len(f)==3
        if coordinates != None:
            assert self.n == len(coordinates)
            for c in coordinates:
                assert len(c)==3
    
    @classmethod
    def fromobj(cls, filename):
        faces, vertices = draw.obj_read(filename)
        return cls(faces, vertices)

    def draw(self):
        draw.draw(self.faces, self.coordinates.tolist())
        

    def angleDefect(self, vertex): # vertex is an integer (vertex index from 0 to self.n-1)
        '''
        При реализации этой функции будем использовать определение из виикипедии:
        Дефект — разность между величиной 2Pi и суммой плоских углов многогранника, примыкающих к одной вершине.
        
        Для этого сначала найдем все грани, где есть наша вершина, а после, для каждой грани посчитаем угол
        нашей вершины в этой грани и вычтем этот угол из 2Pi.
        '''
        
        d = 2 * math.pi
        for face in self.faces:
            for v in face:
                if (v == vertex):
                    neighbors = []
                    for neighbor in face:
                        if (neighbor != vertex):
                            neighbors.append([self.coordinates[neighbor][0] - self.coordinates[vertex][0], self.coordinates[neighbor][1] - self.coordinates[vertex][1], self.coordinates[neighbor][2] - self.coordinates[vertex][2]])
                    cos = (neighbors[0][0] * neighbors[1][0] + neighbors[0][1] * 
                           neighbors[1][1] + neighbors[0][2] * neighbors[1][2]) / (math.sqrt(neighbors[0][0] ** 2 + neighbors[0][1] ** 2 + neighbors[0][2] ** 2) * math.sqrt(neighbors[1][0] ** 2 + neighbors[1][1] ** 2 + neighbors[1][2] ** 2))
                    d -= math.acos(cos)
        return d

    def buildLaplacianOperator(self, anchors = None, anchor_weight = 1.): # anchors is a list of vertex indices, anchor_weight is a positive number
        if anchors == None:
            anchors = []
            
        #Для каждого ребра в треугольнике найдем список соседних по грани вершин
        vertexes = {}
        for face in self.faces:
            v1, v2, v3 = sorted(face)
            vertexes[(v1, v2)] = []
            vertexes[(v2, v3)] = []
            vertexes[(v1, v3)] = []
        for face in self.faces:
            v1, v2, v3 = sorted(face)
            vertexes[(v1, v2)].append(v3)
            vertexes[(v2, v3)].append(v1)
            vertexes[(v1, v3)].append(v2)
        for face in self.faces:
            v1, v2, v3 = sorted(face)
            vertexes[(v1, v2)] = list(set(vertexes[(v1, v2)]))
            vertexes[(v2, v3)] = list(set(vertexes[(v2, v3)]))
            vertexes[(v1, v3)] = list(set(vertexes[(v1, v3)]))
            
        for face in self.faces:
            v1, v2, v3 = sorted(face)
            if len(vertexes[(v1, v2)]) != 2:
                print("Vertex: ", v1, v2, "N: ", vertexes[(v1, v2)])
                
            if len(vertexes[(v2, v3)]) != 2:
                print("Vertex: ", v2, v3, "N: ", vertexes[(v2, v3)])
                
            if len(vertexes[(v1, v3)]) != 2:
                print("Vertex: ", v3, v1, "N: ", vertexes[(v1, v3)])
            
        weight = {}
        for (v1, v2) in vertexes.keys():
            v3 = vertexes[(v1, v2)][0]
            v4 = vertexes[(v1, v2)][1]
            
            u11 = self.coordinates[v1] - self.coordinates[v3]
            u21 = self.coordinates[v2] - self.coordinates[v3]
            a1 = np.arccos(np.dot(u11, u21) / (np.linalg.norm(u11) * np.linalg.norm(u21)))
            
            u12 = self.coordinates[v1] - self.coordinates[v4]
            u22 = self.coordinates[v2] - self.coordinates[v4]
            a2 = np.arccos(np.dot(u12, u22) / (np.linalg.norm(u12) * np.linalg.norm(u22)))
            
            weight[(v1, v2)] = 1/2 * (math.cos(a1) / math.sin(a1) + math.cos(a2) / math.sin(a2))
          
        matrix = [[0] * self.n for i in range(self.n + len(anchors))]
        
        for i in range(self.n):
            matrix[i][i] = -1
        
        #нормируем веса
        norms = np.zeros(self.n)
        for (v1, v2) in weight.keys():
            norms[v1] += weight[(v1, v2)]
            norms[v2] += weight[(v1, v2)]
        
        for v1 in range(self.n):
            for v2 in range(self.n):
                if (v1, v2) in vertexes.keys() and v1 != v2:
                    matrix[v1][v2] = weight[(v1, v2)] / norms[v1]
                    matrix[v2][v1] = weight[(v1, v2)] / norms[v2]
                    
        for i in range(len(anchors)):
            matrix[self.n + i][anchors[i]] = anchor_weight
                    
        return matrix
    
    def smoothen(self, power=1.0): 
        L_None = self.buildLaplacianOperator() 
        self.coordinates = self.coordinates + power * (L_None @ self.coordinates)

    def transform(self, anchors, anchor_coordinates, anchor_weight = 1.):# anchors is a list of vertex indices, anchor_coordinates is a list of same length of vertex coordinates (arrays of length 3), anchor_weight is a positive number
        L_None = np.array(self.buildLaplacianOperator())
        L = np.array(self.buildLaplacianOperator(anchors, anchor_weight))
        x = (L_None @ self.coordinates)
        for i in range(len(anchor_coordinates)):
            for j in range(len(anchor_coordinates[i])):
                anchor_coordinates[i][j] *= anchor_weight
        b = np.vstack((x, anchor_coordinates))
        self.coordinates = np.linalg.inv(L.transpose() @ L) @ L.transpose() @ b

def dragon(): #
    mesh = Mesh.fromobj("dragon.obj")
    mesh.draw()
