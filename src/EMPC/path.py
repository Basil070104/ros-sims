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


import os                       as OS
import numpy                    as NP
import matplotlib.pyplot        as PLT
import torch
from scipy.interpolate          import interp1d

import map                      as Map
from utility                    import *


class Path:
    """
    Generate reference path for MPC
    """

    def __init__(self, args, PATH: str, samplings: int, wnd: int, map) -> None:
        self.args   = args
        self.map    = map
        self.origin = self.args.origin[:-1]
        self.dest   = self.args.dest[:-1]
        self.obst   = self.map.obst
        self.samplings = samplings
        self.wnd    = wnd
        self.PATH   = PATH
        self.FIND   = False

        if OS.path.exists(self.PATH):
            self.ref, self.cost = torch.load(self.PATH)
            self.paths = NP.array(list(zip(*self.ref)))
            self.FIND = True

    def sampling(self, origin: tuple, nodes: list, visited: list, restrict: bool=False, debug: bool=False) -> list:
        """
        Sampling nodes within a range (=delta)

        return:
            @cords:         
            @samplings:     
        """

        idx, xy, dest = origin[0], origin[1], self.dest
        cords, samplings = [], []

        #   [1-1] check if destination can be connected directly (within connection area) -> RETURN DESTINATION as sampled node
        FIND = NP.linalg.norm(NP.array(xy) - NP.array([dest])) <= self.args.delta
        FIND = FIND and not Map.edge_has_collision(NP.array(xy), NP.array(self.dest), self.obst)
        if FIND:
            #   [+] LOG:  find destination within a certain distance around current node
            log(option="syst", msg=f"Find destination within {dest} around [{xy[0]:<19.15f}{xy[1]:^19.15f}]    ->    ID: {idx}", color=True)
            return NP.array([dest]), zip([f"{self.iter}-{len(samplings) + 1}"], [dest])

        #   [1-2] sampling
        while len(samplings) < self.samplings:
            sample = NP.random.rand(2)
            near_node = NP.array(xy)
            next_node = near_node + (sample - near_node) / NP.linalg.norm(sample - near_node) * self.args.delta

            if not ( Map.point_has_collision(next_node, self.obst)              or
                     Map.edge_has_collision(near_node, next_node, self.obst)    or
                     Map.point_validation(next_node.tolist()) ):

                if ( restrict and 
                     not Map.point_has_collision(next_node, NP.array([(self.args.origin[0]-0.1, self.args.origin[1]-0.1, 1-self.args.origin[0], 1-self.args.origin[1])])) ):
                    continue

                cords.append(next_node.tolist())
                samplings.append(f"{self.iter}-{len(samplings) + 1}")

        samplings = list(zip(samplings, cords))
        return NP.array(cords), samplings
    def cost_func(self, cords: list) -> list:
        """
        MIN heuristic distance of each step in the predicted control sequence
        """

        return min(NP.linalg.norm(NP.array(cords) - NP.array(self.dest), axis=1))
    def add_to_graph(self, node, parent_idx, nodes, parent, graph, debug=False):
        if debug: log(option="syst", msg=f"Add node: {node}")

        nodes.append(node)
        parent.append(parent_idx)
        graph = NP.pad(graph, ((0,1), (0,1)))
        graph[parent_idx, -1] = 1
        graph[-1, parent_idx] = 1

        return nodes, parent, graph
    def add_rewire(self, next_node, nodes, graph, parent, cost, obst, neighbor_dist, debug=False):      # AFFECT PERFORMANCE
        """
        Wires to neighbor with lowest cost, rewires neighbors through x_new if doing so decreases their cost
        """

        nodes_array = NP.array(nodes)

        # ===============================================================================
        # NOTE                          AFFECT PERFORMANCE                              =
        # ===============================================================================
        # filter nodes with edge collisions
        all_neighbors = (NP.linalg.norm(nodes_array - next_node, axis=1) < neighbor_dist).nonzero()[0]
        all_neighbors = (NP.linalg.norm(nodes_array - next_node, axis=1) <= neighbor_dist * 1.1).nonzero()[0]   # [NOTE] floating point / loss of precision -> enlarge to fix bugs
        # ===============================================================================

        if debug: log(option="syst", msg=f"Try to add and rewire -> neighbor nodes = {(NP.linalg.norm(nodes_array - next_node, axis=1) <= neighbor_dist).nonzero()}")
        neighbors = NP.array([i for i in all_neighbors if not Map.edge_has_collision(nodes[i], next_node, obst)])
        if len(neighbors) == 0: return False, nodes, graph, parent, cost

        #   [+]log: add and rewire
        if debug: log(option="syst", msg=f"Add and reiwre for node {next_node}")

        #   connect to the node with lowest code in neighbors
        connect_cost = NP.linalg.norm(NP.reshape(nodes_array[neighbors], (-1, len(next_node))) - next_node, axis=1)
        neighbor_cost = NP.array(cost)[neighbors]
        min_cost_idx = NP.argmin(connect_cost + neighbor_cost)
        min_cost_neighbor = neighbors[min_cost_idx]

        #   add to graph
        nodes, parent, graph = self.add_to_graph(next_node, min_cost_neighbor, nodes, parent, graph, debug)
        cost.append(cost[min_cost_neighbor] + connect_cost[min_cost_idx])

        #   rewire neighbors based on next_node if doing so decreases their cost
        rewire_neighbors = (neighbor_cost > connect_cost + cost[-1]).nonzero()[0]
        for neighbor_idx in rewire_neighbors:
            neighbor = neighbors[neighbor_idx]
            cost[neighbor] = connect_cost[neighbor_idx] + cost[-1]
            graph[parent[neighbor], neighbor] = graph[neighbor, parent[neighbor]] = 0
            parent[neighbor] = len(nodes) - 1
            graph[parent[neighbor], neighbor] = graph[neighbor, parent[neighbor]] = 1

        return True, nodes, graph, parent, cost
    def predict(self, idx: int, node: tuple, cost_func, data: list) -> list:
        # ===============================================================================
        # NOTE                          AFFECT PERFORMANCE                              =
        # ===============================================================================
        
        """
        Look ahead to predict future paths with associated costs
        """

        #   (1) copy input data
        _nodes, _graph, _parent, _cost = data
        nodes = _nodes[:]
        graph = NP.array(_graph, copy=True)
        parent = _parent[:]
        cost = _cost[:]

        cur = node[1]
        ids = []
        ctl_seq = []
        FIND = False

        #   (2) randomly generated control sequences to simulate
        while len(ctl_seq) < self.wnd:
            sample = NP.random.rand(2)
            near_node = NP.array(cur)
            next_node = near_node + (sample - near_node) / NP.linalg.norm(sample - near_node) * self.args.delta

            if not (Map.point_has_collision(next_node, self.obst) or Map.edge_has_collision(near_node, next_node, self.obst)):
                ADD, nodes, graph, parent, cost = self.add_rewire(next_node, nodes, graph, parent, cost, self.obst, self.args.delta, debug=False)
                if not ADD: continue

                ids.append(f"{self.iter}-{len(ctl_seq)}")
                ctl_seq.append(next_node)

                # [*]   generated control sequence includes destination -> STOP SAMPLING
                if next_node.tolist() == self.dest:
                    log(option="syst", msg=f"Find destination in the control sequence of node: [{node[1][0]:^20}{node[1][1]:^20}]    ->    ID: {node[0]}", color=True)
                    FIND = True
                    break

                cur = next_node.tolist()

        #   (3) return
        data = zip(ids, ctl_seq)
        path_cost = cost_func(ctl_seq) if not FIND else float("-inf")

        return FIND, path_cost

    def divide_ref(self, path, step_size):
        x, y = path
        step_size = float(self.args.dist / self.args.H)
        points = int(self.cost // step_size)
        alpha = NP.linspace(0, 1, points)
        distance = NP.cumsum(NP.sqrt(NP.ediff1d(x, to_begin=0)**2 + NP.ediff1d(y, to_begin=0)**2))
        distance = distance / distance[-1]
        fx, fy = interp1d(distance, x), interp1d(distance, y)
        x_regular, y_regular = fx(alpha), fy(alpha)
        ref = [x_regular, y_regular]
        return points, ref
    def run(self, restrict: bool=False, debug: bool=False, PLOT: bool=False):
        """
        MPC-based RRT* Algorithm:
            args:
                @origin:    list
                @dest:      list
                @obst:      list

            params:
                @delta:     int (sampling area)
                @dist:      int (connection area)
                @sampling:  int
                @wnd:       int

            return:
                @optimal_path:      tuple
                @visited_path:      tuple

            steps:
                [1]     Graph / Map
                [2]     Sampling
                [3]     Predict
                [4]     Pick MIN cost
                [4-2]   If node cannot be connected (based on RRT*), pick next one
                [4-3]   Move to next node and repeat step 2-1 (until we find the path to reach destination)
                [5]     Find the shortest path and valid path to reach destination
        """

        #   [*] if already generated
        if self.FIND:
            log(option="syst", msg=f"Load refernce path\t\t(FILE: {self.PATH})", color=True)
            self.points, self.ref_div = self.divide_ref(self.ref, float(self.args.dist / self.args.H))  # int, list
            log(option="syst", msg=f"Divide refernce path", color=True)
            if PLOT: self.plot(NP.array(list(zip(*self.ref))))
            return
        # <-----------------------------------------------------------------------------------------------\

        def exec():
            log(option="syst", msg=f"Generate refernce path", color=True)

            prev = None
            cur  = (0, self.origin)
            log(option="syst", msg=f"MPC-based Rapidly-exploring Random Tree (RRT*)", color=True)

            #   [1] Graph / Map
            self.iter = 0
            nodes     = [cur[1]]            # [[x, y]]
            graph     = NP.array([[0]])
            parent    = [None]
            cost_add  = [0]
            path_lib  = dict()              # {PATH_ID: {"source", "dest", "cost"}}
            visited_nodes = [cur[0]]        # idx
            visited_paths = []

            #   (until reach destination)
            while cur[1] != self.dest:
                log(option="path", msg=f"Curr: [{cur[1][0]:<19.15f}{cur[1][1]:>19.15f}]    ID: {cur[0]}")

                #   [2] Sampling / PLOT
                cords, samplings = self.sampling(cur, nodes, visited_nodes, restrict=restrict, debug=debug)
                if PLOT: self.map.plot_nodes(iter=self.iter, cords=cords, origin=cur[1], dest=self.dest)

                #   [3] Predict
                paths = set()
                for idx, node in samplings:
                    PATH_ID = f"{self.iter}-{len(paths)+1}"
                    path = {"source": cur, "dest": (idx, node)}
                    path_lib[PATH_ID] = path
                    paths.add(PATH_ID)

                    data = [nodes, graph, parent, cost_add]
                    FIND, cost = self.predict(idx=PATH_ID, node=(idx, node), cost_func=self.cost_func, data=data)
                    path_lib[PATH_ID]["cost"] = cost

                    if FIND:
                        log(option="path", msg=f"Destination is picked up by the predicted control sequence", color=True)
                        path_lib[PATH_ID]["cost"] = float("-inf")
                        visited_nodes.append("DEST")
                        break

                if debug: log(option="syst", msg=f"Collect {len(paths)} nodes by sampling", color=True)

                """
                # ===============================================================================
                """

                #   [4] Pick MIN cost
                FIND = False
                while not FIND and paths:
                    min_cost_path_id = min({k: path_lib[k] for k in paths}, key=lambda k: path_lib[k]["cost"])
                    next_node = path_lib[min_cost_path_id]["dest"][1]

                    #   [+] DEBUG: sampled nodes
                    if debug:
                        distance = NP.linalg.norm(NP.array(cur[1]) - next_node)
                        valid = distance <= self.args.delta
                        exist = (nodes[-1] == NP.array(cur[1])).all()
                        log(option="syst", msg=f"Node: [{next_node[0]:^20}{next_node[1]:^20}]    Distance: {distance:<25} ({valid:^5}, {exist:^5})")

                    FIND, nodes, graph, parent, cost_add = self.add_rewire(NP.array(next_node), nodes, graph, parent, cost_add, self.obst, self.args.delta, debug=False)

                    #   edge collision -> remove node from path libaray to avoid infinite while loop
                    if not FIND:
                        log(option="syst", msg=f"Retry next node due to edge collision               ID: {min_cost_path_id}", color=True)
                        if PLOT:
                            visited_edges = [*zip(*NP.nonzero(graph))]
                            Map.plot(title=f"Edge collision", map=self.map, origin=cur[1], dest=next_node, nodes=NP.array(nodes), edges=visited_edges)

                        paths.remove(min_cost_path_id)
                        continue

                    #   [+] PLOT: required input data
                    dist = NP.linalg.norm(NP.array(cur[1]) - next_node)
                    visited_nodes.append(path_lib[min_cost_path_id]["dest"][0])
                    visited_paths.append(min_cost_path_id)

                    #   [5] Move
                    prev = cur
                    cur = path_lib[min_cost_path_id]["dest"]
                    self.iter += 1

                    """
                    # ===============================================================================
                    """

                    #   [E-MPC] random scenarious: regenerate map to find reference path
                    if self.iter > 150:
                        cur = (0, self.origin)
                        self.iter = 0
                        nodes     = [cur[1]]
                        graph     = NP.array([[0]])
                        parent    = [None]
                        cost_add  = [0]
                        path_lib  = dict()
                        visited_nodes = [cur[0]]
                        visited_paths = []

                        # self.map, fig = self.map.regen(self.args)
                        # self.obst = self.map.obst
                        break
                        
                    """
                    # ===============================================================================
                    """

                #   [+] LOG / PLOT: current graph
                if PLOT and path_lib:
                    log(option="path", msg=f"Next: [{next_node[0]:^20}{next_node[1]:^20}]    Distance: {dist:.7f}\tCost: {path_lib[min_cost_path_id]['cost']}")

                    visited_edges = [*zip(*NP.nonzero(graph))]
                    Map.plot(title=f"Iteration: {self.iter} / Current Graph (#{len(nodes)})", map=self.map, origin=prev[1], dest=next_node,
                            nodes=NP.array(nodes), edges=visited_edges)

                #   [*] reach destination / no valid paths -> redo sampling
                if "DEST" in visited_nodes: log(option="path", msg=f"Reach Destination: {self.dest}", color=True); break
                if not FIND: log(option="path", msg=f"No path can take to proceed -> Stay at current position and re-do sampling", color=True)

            #   [+] LOG: reach destination
            if debug: log(option="path", msg=f"Dest: {self.dest}", color=True)

            """
            # ===============================================================================
            """

            #   [6] Shortest path / Visited path
            FIND, nodes, graph, parent, cost_add = self.add_rewire(NP.array(self.dest), nodes, graph, parent, cost_add, self.obst, self.args.delta, debug=debug)
            path, cost = [], 0.0
            if nodes[-1].tolist() == self.dest:
                idx = len(nodes) - 1
                while parent[idx] is not None:
                    cost += NP.linalg.norm(nodes[idx] - nodes[parent[idx]])
                    path.insert(0, list(nodes[idx]))
                    idx = parent[idx]
            path = NP.array([self.origin] + path)

            visited_cost, tmp = 0, []
            for element in visited_paths:
                start, dest = path_lib[element]["source"], path_lib[element]["dest"]
                length = NP.linalg.norm(NP.array(start[1]) - NP.array(self.dest))
                tmp.append(start[1])
                visited_cost += length
            visited_paths = NP.array(tmp + [self.dest])

            return path, cost

            """
            # ===============================================================================
            """
        path, cost = exec()

        # <-----------------------------------------------------------------------------------------------/
        #   [+] PLOT: result
        if PLOT: self.plot(path)

        #   [1] save results
        if self.args.map == "random": self.PATH = f"./pth/{self.map.seed}.pth"
        if not restrict:
            log(option="syst", msg=f"Save reference path: {self.PATH}", color=True, pattern=Colors.Pattern(text=Colors.TRED))
            torch.save(([path[:, 0].tolist(), path[:, 1].tolist()], cost), self.PATH)

        #   [2] FOR MPC
        self.ref = [path[:, 0].tolist(), path[:, 1].tolist()]
        self.paths = NP.array(list(zip(*self.ref)))
        self.cost = cost

        #   [2] divide reference path
        self.points, self.ref_div = self.divide_ref(self.ref, float(self.args.dist / self.args.H))  # int, list
        log(option="syst", msg=f"Divide refernce path", color=True)
        
    def plot(self, path):
        fig = self.map.get_plot(title="Result", origin=self.origin, dest=self.dest)
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
        PLT.figure(fig)
        PLT.gca().axes.xaxis.set_ticks([])
        PLT.gca().axes.yaxis.set_ticks([])
        PLT.plot(path[:, 0], path[:, 1], marker='o', color='b', label="Reference Path", linewidth=3)
        PLT.legend()
        return fig