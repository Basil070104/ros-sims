#######################################################################
# Copyright (C) 2024 Yuan-Yao Lou (yuanyao.lou@gmail.com)             #
# Permission given to modify or distribute the code as long as you    #
# keep this declaration at the top.                                   #
#######################################################################

"""
Author  : Yuan-Yao Lou (Mike)
Title   : PhD student in ECE at Purdue University
Email   : yylou@purdue.edu
Website : https://yylou.github.io/
Date    : 2024/04/01

Project : Edge-assisted MPC
Version : v2.1
"""

#  [Reference]
#  - https://matplotlib.org/stable/api/markers_api.html
#  - https://matplotlib.org/stable/gallery/color/named_colors.html


import numpy                    as NP
import os                       as OS
import random                   as RD

import matplotlib.pyplot        as PLT
from matplotlib.patches         import Rectangle
from matplotlib.collections     import PatchCollection
# from skopt.sampler              import Hammersly
from sklearn.neighbors          import kneighbors_graph as knn_graph
from scipy.spatial              import Voronoi, voronoi_plot_2d
from shapely.geometry.polygon   import Polygon
from shapely.geometry           import Point
from shapely.geometry           import mapping
import torch

from EMPC.utility                    import log


class Map():
    def __init__(self, args):
        """
        obst:               [[x, y, width, height], ...]
        origin, dest:       [x, y]
        nodes:              [[x, y], ...]
        edges:              [(indx p1, indx p2), ...]
        """

        self.args = args

        #   (1-1) map settings
        self.origin = args.origin[:-1]
        self.dest   = args.dest[:-1]
        self.n_obst = args.obst
        self.random_seed = int(args.map) if int(args.map) != -1 else RD.randint(0, 1000)
        
        #   (1-2) initialization
        self.obst, self.seed = self.build_map(self.random_seed)
        self.cost_area  = False
        self.edge_nodes = None
        
        #   (2) costly area
        if   args.map != -1 and args.map == 130 and args.sensing:
            self.cost_area   = [([0.48759, 0.56327], 0.06305, 0.07134), ([0.76092, 0.73501], 0.10108, 0.10316)]
        elif args.map != -1 and args.map == 139 and args.sensing:
            self.cost_area   = [([0.51759, 0.66327], 0.06305, 0.07134), ([0.76092, 0.73501], 0.10108, 0.10316)]
        elif args.map != -1 and args.map == 2837 and args.sensing:
            self.cost_area   = [([0.4, 0.55], 0.1, 0.07), ([0.4, 0.55], 0.1, 0.07)]
        elif args.sensing:
            self.cost_area   = [([0.6, 0.6], 0.1, 0.1), ([0.7, 0.7], 0.1, 0.1)]

        #   (3-1) edge nodes
        self.vor, self.edge_nodes = self.gen_edge_server()

        #   (3-2) random nodes
        self.n_nodes    = args.nodes * 10 if args.nodes != 0 else 25
        self.size       = 10 * 10
        # self.nodes      = NP.array(Hammersly().generate([(0, self.size), (0, self.size)], self.n_nodes)) / self.size
        self.nodes      = NP.random.uniform(0, 1, (self.n_nodes, 2))
        self.all_nodes  = NP.vstack([self.origin, self.dest, self.nodes])
        valid_indx      = [i for i in range(len(self.all_nodes)) if not point_has_collision(self.all_nodes[i,:], self.obst)]
        self.nodes      = self.all_nodes[valid_indx,:]
        self.n_neighbor = 5
        graph = knn_graph(self.nodes, self.n_neighbor, mode='connectivity', include_self=False)
        self.graph = self.eliminate_edges(graph, self.nodes, self.obst)
        self.edges = [*zip(*NP.nonzero(self.graph))]

        log(option="syst", msg=f"Map initialization ({args.map} -> {self.seed})")

    def regen(self, args):
        self.__init__(args)
        return self, self.get_plot(title="Map", origin=args.origin, dest=args.dest)

    # ========================================================================================================= #

    def build_map(self, seed):
        self.valid = False
        while not self.valid:
            NP.random.seed(seed)        
            obst = NP.random.rand(self.n_obst, 4)
            obst[:,2:] = .8 * (obst[:,2:] + .3) / NP.sqrt(self.n_obst) # scale width and height
            self.valid = not (point_has_collision(self.origin, obst) or point_has_collision(self.dest, obst))
            seed += 100

        return obst, seed

    def get_obst_dist(self, cord: list, FILTER: int):
        #   | Array: center
        dist = 0.0
        for obst in self.edge_nodes[FILTER]["obstacles"]:
            central = (obst[0] + obst[2] / 2, obst[1] + obst[3] / 2)
            dist += NP.linalg.norm(NP.array(central) - NP.array(cord))
        return dist

    def eliminate_edges(self, _graph, nodes, obst):
        """
        Given connectivity matrix (graph) G, return new matrix with bad edges removed
        """

        edges = [*zip(*NP.nonzero(_graph))]
        graph = NP.zeros(_graph.shape)
        for edge in edges:
            if not edge_has_collision(nodes[edge[0]], nodes[edge[1]], obst):
                graph[edge[0], edge[1]] = 1
        
        return graph

    def gen_edge_server(self):
        n_nodes = self.args.nodes or 10

        #   (1) generate coordinates of edge servers
        nodes = []
        while len(nodes) < n_nodes:
            data = NP.random.uniform(size=(2))
            if not point_validation(data) and not point_has_collision(data, self.obst):
                nodes.append(data)
        
        #   (2) coverages
        vor = Voronoi(nodes)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        polygons = []
        for region in regions:
            polygons.append([])
            for idx in region: polygons[-1].append(vertices[idx])

        #   (3) node encapsulation
        edge_nodes = {}
        for polygon in polygons:
            polygon = Polygon(polygon)
            
            #   (3-1) check intersected obstacles
            obstacles = []
            for obst_array in self.obst:
                _p1 = Point(obst_array[0], obst_array[1])
                _p2 = Point(obst_array[0] + obst_array[2], obst_array[1])
                _p3 = Point(obst_array[0], obst_array[1] + obst_array[3])
                _p4 = Point(obst_array[0] + obst_array[2], obst_array[1] + obst_array[3])
                if polygon.contains(_p1) or polygon.contains(_p2) or polygon.contains(_p3) or polygon.contains(_p4):
                    obstacles.append(obst_array)

            #   (3-2) map the coverage with server
            for node in nodes:
                _p = Point(node[0], node[1])
                if polygon.contains(_p):
                    edge_nodes[len(edge_nodes)] = {"location": (node[0], node[1]),
                                                   "region": polygon,
                                                   "obstacles": NP.array(obstacles)}
        return vor, edge_nodes

    def get_obst_collection(self, obst):
        return PatchCollection([Rectangle(xy=obst[i,:2], width=obst[i,2], height=obst[i,3])
                                    for i in range(self.n_obst)],
                                    facecolor='black')

    # ========================================================================================================= #

    def plot_task(self, origin, goal, obst):
        #   (1) Obstacles
        obst_collection = self.get_obst_collection(obst)                        
        PLT.gca().add_collection(obst_collection)
        PLT.gca().set(xlim=(0, 1), ylim=(0, 1))

        #   (2) Start / End points
        PLT.scatter(origin[0], origin[1], c='g', marker='x', label='Start', zorder=999)
        PLT.scatter(goal[0], goal[1],     c='r', marker='x', label='Destination', zorder=999)

        #   (3) High-cost areas
        if self.cost_area:
            PLT.gca().add_collection(PatchCollection([Rectangle(xy=array[0],
                                                                width=array[1],
                                                                height=array[2]) for array in self.cost_area],
                                                                color='none',
                                                                edgecolor='#808080',
                                                                hatch='xx', zorder=-1))
            PLT.scatter(10, 10, c='#808080', marker='x', s=50, label='Costly Area')

    def plot_graph(self, nodes, edges, node_color='grey', edge_color='grey', opacity=1, linewidth=.5):
        #   (4) Randomly distributed nodes
        if edge_color is not None:  
            for edge in edges:  
                x, y = (nodes[edge[0]][0], nodes[edge[1]][0]), (nodes[edge[0]][1], nodes[edge[1]][1])
                PLT.plot(x, y, linewidth=linewidth, c=edge_color, alpha=opacity)
        if node_color is not None:
            PLT.scatter(nodes[:,0], nodes[:,1], s=20, c=node_color, alpha=opacity, label="?")

        #   (5) Edge servers
        if self.edge_nodes:
            nodes = NP.array([self.edge_nodes[key]["location"] for key in self.edge_nodes])
            PLT.scatter(nodes[:,0], nodes[:,1], marker='H', s=100, c='red', alpha=0.2, label=f"Server")
            for polygon in self.edge_nodes.values():
                PLT.plot(polygon["region"].exterior.xy[0], polygon["region"].exterior.xy[1], c='grey', alpha=0.15, zorder=-1)

    def get_plot(self, title, origin, dest):
        fig = PLT.figure(title, figsize=(3.7, 3.7), clear=True)
        self.plot_task(origin, dest, self.obst)
        if self.args.nodes: self.plot_graph(self.nodes, self.edges, edge_color=None, node_color=None, opacity=.3)
        return fig

    def plot_nodes(self, iter, cords, origin, dest):
        fig = self.get_plot(title=f"Iteration: {iter}", origin=origin, dest=dest)
        PLT.figure(fig)
        PLT.scatter(cords[:,:1], cords[:,1:2])
        fig.show()
        input()

    # ========================================================================================================= #

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Reference
    ---------
    https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    #   Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    #   Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            #   finite region
            new_regions.append(vertices)
            continue

        #   reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                #   finite ridge: already in the region
                continue

            #   Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= NP.linalg.norm(t)
            n = NP.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = NP.sign(NP.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        #   sort region counterclockwise
        vs = NP.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = NP.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = NP.array(new_region)[NP.argsort(angles)]

        #   finish
        new_regions.append(new_region.tolist())

    return new_regions, NP.asarray(new_vertices)

def plot(title: str, map, origin, dest, nodes, edges=None):
    PLT.close("all")
    fig = PLT.figure(title, figsize=(3.7, 3.7), clear=True)
    obst_collection = PatchCollection(
        [Rectangle(
            xy=map.obst[i,:2],width=map.obst[i,2],
            height=map.obst[i,3]) for i in range(map.n_obst)]
        , facecolor='black')
    PLT.gca().add_collection(obst_collection)
    PLT.gca().set(xlim=(0, 1), ylim=(0, 1))
    PLT.scatter(origin[0], origin[1], c='g', marker='x', label='Start', zorder=997)
    PLT.scatter(dest[0],   dest[1],   c='r', marker='x', label='Destination', zorder=997)

    if map.cost_area:
        PLT.gca().add_collection(PatchCollection([Rectangle(xy=array[0],
                                                            width=array[1],
                                                            height=array[2]) for array in map.cost_area],
                                                            color='none',
                                                            edgecolor='#808080',
                                                            hatch='xxxx'))
        PLT.scatter(10, 10, c='#808080', marker='x', s=50, label='Costly Area')
    
    if edges:
        for edge in edges:  
            x, y = (nodes[edge[0]][0], nodes[edge[1]][0]), (nodes[edge[0]][1], nodes[edge[1]][1])
            PLT.plot(x, y, linewidth=.5, c="red", alpha=1)

    PLT.scatter(nodes[:,0], nodes[:,1], s=20, c="grey", alpha=1)
    fig.show()
    input()

def point_validation(point):
    """
    Check for any out-of-bound point
    """

    return not (1 > point[0] > 0) or not (1 > point[1] > 0)

def point_has_collision(point, obst):
    """
    Check for any obstacle collision
        point=[x, y],
        obst=[[x, y, width, height], ...]
    requires o_x < p_x < (o_x + o_w) and o_y < p_y < (o_y + o_h)
    """
    point_intersect = (  (point[0] >= obst[:,0])
                       * (point[0] <= (obst[:,0] + obst[:,2])) 
                       * (point[1] >= obst[:,1])
                       * (point[1] <= (obst[:,1] + obst[:,3])))
    return point_intersect.any()

def edge_has_collision(p0, p1, obst):
    return edge_intersect(p0, p1, obst).any()

def edge_intersect(p0, p1, obst):
    """
    Check if edge cuts through any obst in obst
    p0, p1 = [x, y] obst = [[x, y, width, height], ...]

    For a vector Z and point q, we can determine whether q is on the left or
    right of Z by forming Z_perp = [Z[1], -Z[0]] and looking at the sign of the
    the inner product <Z_perp, q>. If Z is centered at z0, we instead use
    sign(<Z_perp, q-z>)

    Now consider vector p = p1 - p0, and corners of obst c0, c1, c2, c3
    For collision we require corners on different sides of p, and that p0 and p1
    be on opposite sides of two planes of obst. Since those planes are vertical/
    horizontal in this case, we don't have to mess with inner products
    """

    p0, p1 = NP.array(p0), NP.array(p1)
    p = p1 - p0
    p_perp = [p[1], -p[0]]
    c0 = obst[:,:2]                                                             # bottom left
    c1 = NP.vstack([obst[:,0] + obst[:,2], obst[:,1]]).T                        # bottom right
    c2 = NP.vstack([obst[:,0] + obst[:,2], obst[:,1] + obst[:,3]]).T            # top right
    c3 = NP.vstack([obst[:,0], obst[:,1] + obst[:,3]]).T                        # top left
    corner_split = NP.abs(NP.sign(NP.inner(p_perp, c0 - p0)) + NP.sign(NP.inner(p_perp, c1 - p0))
                        + NP.sign(NP.inner(p_perp, c2 - p0)) + NP.sign(NP.inner(p_perp, c3 - p0))) < 4
    plane_intersect = NP.sum(NP.abs(NP.sign(p0 - c0)-NP.sign(p1 - c0))
                            +NP.abs(NP.sign(p0 - c2)-NP.sign(p1 - c2)),axis=1)>=4
    return (corner_split * plane_intersect)


# wrapper
def init(args):
    PATH = f"maps/{args.map}-{args.nodes}-{args.sensing}.map"
    
    if args.map != -1 and OS.path.exists(PATH):
        log(option="empc", msg=f"Load map\t(FILE: {PATH})", color=True)
        return torch.load(PATH)

    log(option="syst", msg=f"Map builder ({args.map})")
    map = Map(args)
    plot = map.get_plot(title="Map Initialization", origin=map.origin, dest=map.dest)
    
    PATH = f"maps/{map.random_seed}-{args.nodes}-{args.sensing}.map"
    torch.save((map, plot), PATH)
    log(option="empc", msg=f"Save map\t(FILE: {PATH})", color=True)

    return map, plot