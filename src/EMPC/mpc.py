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


from collections                import Counter
from collections                import defaultdict
from copy                       import deepcopy as Copy
import datetime                 as DATE
import glob                     as GLOB
import json                     as JSON
import math                     as MATH
import numpy                    as NP
import matplotlib.pyplot        as PLT
from multiprocessing            import Process, Manager
import random                   as RD
import sys                      as SYS

from matplotlib.patches         import Rectangle
from matplotlib.collections     import PatchCollection
import torch
import similaritymeasures
from scipy.interpolate          import interp1d
from scipy.stats                import truncnorm
from shapely.geometry           import Point
from shapely.geometry           import mapping

import map                      as Map
import utility


class Agent:

    NCTRL = 2
    NPOS  = 3

    def __init__(self, args, num, map_obj, path, motion, ctrls, dtype, SEARCH: bool=False) -> None:
        self.args   = args
        self.num    = num
        self.map    = map_obj
        self.origin = self.args.origin
        self.dest   = self.args.dest[:-1]
        self.path   = path
        self.points = self.path.points
        self.ref    = self.path.ref_div
        self.motion = motion
        self.ctrls  = ctrls
        self.dtype  = dtype

        # ---
        # if self.args.plot: 
        #     self.plot_map(map=self.map)
        #     self.plot_ref(ref=self.path.ref, TRIM=True)

        #   |1|  Place starting point
        if self.args.search and SEARCH:
            self.origin, traj = self.calibrate()
            utility.log(option="empc", msg=f"Place starting point at {self.origin}")

    """ System """
    def calibrate(self, origin=None, iterIdx=None):
        def get_route(poses: 'torch.Tensor'):
            """
            Extract routing paths (x- and y-axis coordinates) from a set of 3D poses

            args:
                @poses:     Tensor (3-dimension tensor, [x, y, steering])

            return:
                @lines[0]:  x coordinates
                @lines[1]:  y coordinates
                @lines[2]:  steering
            """

            lines = []
            for line in poses[:, :]:
                lines.append([line[:, 0].tolist(), line[:, 1].tolist(), line[:, 2].tolist()])

            return lines

        steps = self.args.H
        MIN_cost = float("inf")
        traj = None
        if origin  is None: origin  = self.args.origin
        if iterIdx is None: iterIdx = 0
        ref = [self.ref[0][iterIdx:min(self.points,iterIdx+steps)], self.ref[1][iterIdx:min(self.points,iterIdx+steps)]]

        for angle in range(-500, 500):
            zero = self.ctrls[:1]                                   # [1, H, NCTRL]
            poses = self.dtype(1, zero.shape[1], 3).zero_()         # [1, H, NPOS]
            poses[:] = self.dtype([origin[0], origin[1], angle/100.0])
            self.motion.set(1)

            for t in range(1, zero.shape[1]):
                cur_pose = poses[:, t - 1]
                cur_ctrl = zero[:, t - 1]
                poses[:, t] = self.motion.apply(cur_pose, cur_ctrl)

            pred = get_route(poses)
            for idx, path in enumerate(pred):
                path = NP.array(path)[:, :min(self.points-iterIdx,steps)]
                xy = [NP.array(path[0]), NP.array(path[1])]
                cost_frechet = similaritymeasures.frechet_dist(xy, ref)

                if cost_frechet < MIN_cost:
                    origin = [origin[0],origin[1],angle/100.0]
                    MIN_cost = cost_frechet
                    traj = pred[idx]

            self.motion.set(self.args.N)

        return origin, self.dtype([list(zip(*traj))])
    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        """
        Ex: 
            # generate 3 data-points
            distribution = self.get_truncated_normal(mean=5, sd=5, low=0, upp=30)
            dp = distribution.rvs(3)

            # round up to integer
            NP.rint(dp)
        """
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    def show(self, iter=None, cost=None, dist=None, assist=None, blind=None, traj=None, cp=None):
        CLR = utility.Colors.Pattern(text=utility.Colors.TYELLOW)
        END = utility.Colors.END
        print("\n")
        print(f"\t{'X':<25}{'Y':<25}{'Steering':<25}{'Trajectory'}")
        print("\t" + "="*125)
        print(f"\t{self.cur[0]:<25.15f}{self.cur[1]:<25.15f}{self.cur[2]:<25.15f}{CLR}{traj}{END}")
        print()
        print(f"\t{'Edge Assistance':<25}{'Computing Capability'}")
        print("\t" + "="*125)
        print(f"\t{str(assist):<25}{CLR}{cp}{END}")
        print()
        print(f"\t{'Server Distance':<25}{'Connections'}")
        print("\t" + "="*125)
        print(f"\t{CLR}{str(dist):<25}{self.connections}{END}")
        print()
        print(f"\t{'Iteration':<25}{'Total Cost':<25}{'Blind Spots'}")
        print("\t" + "="*125)
        print(f"\t{CLR}{iter:<25}{cost:<25.10f}{len(blind):<25}{END}")
        print("\n")

    """ Visual """
    def plot_map(self, map=None, TRIM: bool=False):
        PLT.close('all')
        if map: map = map
        else: map = self.ref

        #   (1) MAP
        fig = self.map.get_plot(title="Map", origin=self.args.origin, dest=self.args.dest)
        fig.subplots_adjust(top=0.99, bottom=0.16, left=0.01, right=0.99)
        PLT.figure(fig)

        #   | Coverage
        EDGE_ID = NP.argmax([info["region"].contains(Point(self.origin[0], self.origin[1])) for info in self.map.edge_nodes.values()])
        info = self.map.edge_nodes[EDGE_ID]
        PLT.plot(info["region"].exterior.xy[0], info["region"].exterior.xy[1], zorder=0, c='#C41E3A', linewidth=2, label="Area")
        
        #   | Filtered Obstacles
        obst = info["obstacles"]
        obst_collection = PatchCollection([Rectangle(xy=obst[i,:2], width=obst[i,2], height=obst[i,3]) for i in range(len(obst))], facecolor='black', edgecolor='white', hatch="/")
        PLT.gca().add_collection(obst_collection)

        #   (*) TRIM
        if TRIM:
            PLT.xlim([min(self.ref[0])*0.95, max(self.ref[0])*1.05])
            PLT.ylim([min(self.ref[1])*0.95, max(self.ref[1])*1.05])

        #   (*) LEGEND
        PLT.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.005))
        
        #   (*) SHOW
        PLT.gca().axes.xaxis.set_ticks([])
        PLT.gca().axes.yaxis.set_ticks([])
        fig.show()
        input()
    def plot_ref(self, ref=None, TRIM: bool=False):
        PLT.close('all')
        if ref: ref = ref
        else: ref = self.ref

        #   (1) MAP
        fig = self.map.get_plot(title="Reference Path", origin=self.args.origin, dest=self.args.dest)
        fig.subplots_adjust(top=0.99, bottom=0.16, left=0.01, right=0.99)
        
        #   (2) PATH
        PLT.plot(ref[0], ref[1], marker='o', label="Reference Path")

        #   (*) TRIM
        if TRIM:
            PLT.xlim([min(ref[0])*0.95, max(ref[0])*1.05])
            PLT.ylim([min(ref[1])*0.95, max(ref[1])*1.05])

        #   (*) LEGEND
        PLT.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.005))
        
        #   (*) SHOW
        PLT.gca().axes.xaxis.set_ticks([])
        PLT.gca().axes.yaxis.set_ticks([])
        fig.show()
        input()
    def plot_traj(self, map=None, ref=None, idx=None, traj=None, TRIM: bool=True):
        PLT.close('all')
        
        #   (1) MAP
        fig = map.get_plot(title="Predictive Trajectory", origin=self.args.origin, dest=self.args.dest)
        PLT.figure(fig)
        #   (2) PATH
        PLT.plot(ref[0], ref[1], c='#7393B3', alpha=0.3, marker='o', linewidth=3, markersize=3, label="Reference Path")
        #   (3) TRAJ
        MIN_x = MIN_y = float("inf")
        MAX_x = MAX_y = float("-inf")
        for idx, sampling in zip(idx, traj):
            MIN_x, MIN_y = min(MIN_x, torch.min(sampling[:,0])), min(MIN_y, torch.min(sampling[:,1]))
            MAX_x, MAX_y = max(MAX_x, torch.max(sampling[:,0])), max(MAX_y, torch.max(sampling[:,1]))
            PLT.plot(sampling[:,0], sampling[:,1], c="#F28C28", alpha=0.3, linewidth=2, label="Onboard", zorder=999)
            PLT.annotate('{}'.format(idx),
                xy=(sampling[-1][0], sampling[-1][1]), xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points", ha='center', va='bottom', c="#F28C28", zorder=999)

        #   (*) TRIM
        if TRIM:
            PLT.xlim([MIN_x*0.99, MAX_x*1.01])
            PLT.ylim([MIN_y*0.99, MAX_y*1.01])

        #   (*) LEGEND
        handles, labels = PLT.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        del by_label["Destination"]
        PLT.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1))

        #   (*) SHOW
        fig.show()
        input()

    """ Core """
    def move(self, poses, ctrls, horizon):
        for t in range(1, horizon):
            cur_pose = poses[:, t - 1]
            cur_ctrl = ctrls[:, t - 1]
            poses[:, t] = self.motion.apply(cur_pose, cur_ctrl)
        return poses
    def latency(self, OBST: int, DIST: float) -> int:
        if   self.args.latency == 0: threshold = 0.40
        elif self.args.latency == 1: threshold = 0.40
        elif self.args.latency == 2: threshold = 0.20
        elif self.args.latency == 3: threshold = 0.20

        if   DIST > threshold: return None, -1

        #   (mmWave)
        if OBST < 2:
            if       OBST and DIST < (threshold * 0.50): return "mmWave", 3
            elif not OBST and DIST > (threshold * 0.50): return "mmWave", 2
            elif not OBST and DIST < (threshold * 0.50): return "mmWave", 1
            elif not OBST and DIST < (threshold * 0.25): return "mmWave", 0

        #   (mid-band)
        if OBST < 5:
            if   DIST > (threshold * 0.75): return "mid-band", 3
            elif DIST > (threshold * 0.50): return "mid-band", 2
            elif DIST > (threshold * 0.25): return "mid-band", 1
            elif DIST < (threshold * 0.25): return "mid-band", 0
        
        return None, -1
    def connect(self, cur: list, CP: list):
        if   self.args.latency == 0: 
            noise = {
                "mmWave":   [0.91, 0.76, 0.63, 0.50],   # 0.9 ms, 2.4 ms, 3.7 ms, 5.0 ms
                "mid-band": [0.77, 0.38, 0.12, 0.00],   # 2.3 ms, 6.2 ms, 8.8 ms, 11.3 ms
            }
        elif self.args.latency == 1: 
            noise = {
                "mmWave":   [0.95, 0.85, 0.79, 0.72],   # 0.5 ms, 1.5 ms, 2.1 ms, 2.8 ms
                "mid-band": [0.89, 0.63, 0.39, 0.13],   # 1.1 ms, 3.7 ms, 6.1 ms, 8.7 ms
            }
        elif self.args.latency == 2: 
            noise = {
                "mmWave":   [0.91, 0.76, 0.63, 0.50],   # 0.9 ms, 2.4 ms, 3.7 ms, 5.0 ms
                "mid-band": [0.77, 0.38, 0.12, 0.00],   # 2.3 ms, 6.2 ms, 8.8 ms, 11.3 ms
            }
        elif self.args.latency == 3: 
            noise = {
                "mmWave":   [0.95, 0.85, 0.79, 0.72],   # 0.5 ms, 1.5 ms, 2.1 ms, 2.8 ms
                "mid-band": [0.89, 0.63, 0.39, 0.13],   # 1.1 ms, 3.7 ms, 6.1 ms, 8.7 ms
            }
        
        prob = [[0.80, 0.15, 0.04, 0.01],
                [0.15, 0.80, 0.04, 0.01],
                [0.04, 0.15, 0.80, 0.01],
                [0.01, 0.04, 0.15, 0.80]]
        
        EDGE_ID = NP.argmax([info["region"].contains(Point(cur[0], cur[1])) for info in self.map.edge_nodes.values()])
        connections = {0: set(), 1: set(), 2: set(), 3: set()}
        conn_dist = 0.0
        server_comp = {}

        if self.args.nodes:
            for idx, server in self.map.edge_nodes.items():
                location, region = server["location"], server["region"]
                BLOCK = Map.edge_intersect(NP.array(cur[:-1]), NP.array(location), self.map.obst)
                DIST = NP.linalg.norm(NP.array(cur[:-1]) - NP.array(location))
                OBST = sum(BLOCK)

                tech, level = self.latency(OBST, DIST)
                if level != -1:
                    connections[level].add(idx)
                    conn_dist += DIST
                    server_comp[idx] = int(NP.rint(NP.random.choice(noise[tech], p=prob[level]) * CP[idx]))

        return EDGE_ID, connections, conn_dist, server_comp
    def _connect(self, cur: list, CP: list):
        if   self.args.latency == 0: 
            noise = {
                "mmWave":   [0.91, 0.76, 0.63, 0.50],   # 0.9 ms, 2.4 ms, 3.7 ms, 5.0 ms
                "mid-band": [0.77, 0.38, 0.12, 0.00],   # 2.3 ms, 6.2 ms, 8.8 ms, 11.3 ms
            }
        elif self.args.latency == 1: 
            noise = {
                "mmWave":   [0.95, 0.85, 0.79, 0.72],   # 0.5 ms, 1.5 ms, 2.1 ms, 2.8 ms
                "mid-band": [0.89, 0.63, 0.39, 0.13],   # 1.1 ms, 3.7 ms, 6.1 ms, 8.7 ms
            }
        elif self.args.latency == 2: 
            noise = {
                "mmWave":   [0.91, 0.76, 0.63, 0.50],   # 0.9 ms, 2.4 ms, 3.7 ms, 5.0 ms
                "mid-band": [0.77, 0.38, 0.12, 0.00],   # 2.3 ms, 6.2 ms, 8.8 ms, 11.3 ms
            }
        elif self.args.latency == 3: 
            noise = {
                "mmWave":   [0.95, 0.85, 0.79, 0.72],   # 0.5 ms, 1.5 ms, 2.1 ms, 2.8 ms
                "mid-band": [0.89, 0.63, 0.39, 0.13],   # 1.1 ms, 3.7 ms, 6.1 ms, 8.7 ms
            }
        
        prob = [[0.80, 0.15, 0.04, 0.01],
                [0.15, 0.80, 0.04, 0.01],
                [0.04, 0.15, 0.80, 0.01],
                [0.01, 0.04, 0.15, 0.80]]
        
        EDGE_ID = NP.argmax([info["region"].contains(Point(cur[0], cur[1])) for info in self.map.edge_nodes.values()])
        connections = {0: set(), 1: set(), 2: set(), 3: set()}
        conn_dist = 0.0
        server_comp = {}

        if self.args.nodes:
            for idx, server in self.map.edge_nodes.items():
                location, region = server["location"], server["region"]
                BLOCK = Map.edge_intersect(NP.array(cur[:-1]), NP.array(location), self.map.obst)
                DIST = NP.linalg.norm(NP.array(cur[:-1]) - NP.array(location))
                OBST = sum(BLOCK)

                tech, level = self.latency(OBST, DIST)
                if level != -1:
                    connections[level].add(idx)
                    conn_dist += DIST
                    server_comp[idx] = NP.random.choice(noise[tech], p=prob[level]) * 10

        return EDGE_ID, connections, conn_dist, server_comp

    """ Algo """
    def exec(self, cur: list, DEBUG: bool=False, PLOT: bool=False):
        #  Variables
        ctrls = self.ctrls
        num   = self.num
        H     = self.args.H

        """
        # ===============================================================================
        #   Sampling                                                                    =
        # ===============================================================================
        """
        idx  = torch.randperm(ctrls.shape[0])
        traj = ctrls[idx].view(ctrls.size())[:num]
        idx  = idx[:num]

        """
        # ===============================================================================
        #   Predict                                                                     =
        # ===============================================================================
        """
        n_path  = len(traj)
        pred    = self.dtype(n_path, H, self.NPOS).zero_()
        pred[:] = self.dtype(cur)
        self.motion.set(size=num)
        self.pred = self.move(poses=pred, ctrls=traj, horizon=H)

        # ---
        # self.plot_traj(map=self.map, ref=self.ref, idx=idx, traj=pred, TRIM=True)

        self.idx, self.traj = idx, traj
        return self.idx, self.traj, self.pred
    def eval(self, steps=None, ref=None, pred=None):
        pred = NP.array(pred)[:steps, :]                # [[x, y, theta], ...]
        x, y, steering = list(zip(*pred))               # [x1, x2, ...], [theta1, theta2, ...]
        xy, cords = [x, y], list(zip(x, y))             # [x, y], [[x, y], ...]
        func_within = lambda x, y: y[0] > x[0][0] and y[0] < x[0][0]+x[1] and y[1] > x[0][1] and y[1] < x[0][1]+x[2]

        costs = {"Frechet": 0.0, "Destination": 0.0, "Obstacles": 0.0, "Controls": 0.0, "Collision": 0, "Disturbance": 0.0}
        costs["Frechet"] = similaritymeasures.frechet_dist(xy, ref)
        costs["Destination"] = NP.linalg.norm(NP.array(self.dest) - NP.array(cords[-1]))
        costs["Controls"] = float(NP.sum(NP.abs(NP.diff(pred[:,-1]))))
        tmp = self.map.get_obst_dist(cord=cords[len(cords) // 2], FILTER=self.EDGE_ID)
        costs["Obstacles"] = 1 / tmp if tmp else 0.0
        costs["Collision"] = sum([1/i if Map.point_has_collision(cords[i], self.map.obst) else 0 for i in range(1, len(cords))])
        if self.map.cost_area: costs["Disturbance"] = int(sum([int(func_within(self.map.cost_area[0], cur) or func_within(self.map.cost_area[1], cur)) for cur in cords]))

        return costs

class Cost:
    def __init__(self, weight) -> None:
        self.idx = {}
        self.source = {}
        self.Frechet_range   = [float("-inf"), float("inf")]
        self.Dest_range      = [float("-inf"), float("inf")]
        self.Control_range   = [float("-inf"), float("inf")]
        self.Obst_range      = [float("-inf"), float("inf")]
        self.Collision_range = [float("-inf"), float("inf")]
        self.Dist_range      = [float("-inf"), float("inf")]
        self.weight = weight
    def add(self, id: int, data: dict, src: str) -> None:
        self.idx[id] = data
        self.source[id] = src
        for k, v in data.items():
            if   k == "Frechet":     self.Frechet_range   = [max(self.Frechet_range[0], v),   min(self.Frechet_range[1], v)]
            elif k == "Destination": self.Dest_range      = [max(self.Dest_range[0], v),      min(self.Dest_range[1], v)]
            elif k == "Controls":    self.Control_range   = [max(self.Control_range[0], v),   min(self.Control_range[1], v)]
            elif k == "Obstacles":   self.Obst_range      = [max(self.Obst_range[0], v),      min(self.Obst_range[1], v)]
            elif k == "Collision":   self.Collision_range = [max(self.Collision_range[0], v), min(self.Collision_range[1], v)]
            elif k == "Disturbance": self.Dist_range      = [max(self.Dist_range[0], v),      min(self.Dist_range[1], v)]
    def eval(self, AVOID: bool=False) -> None:
        MIN_ID, MIN_COST, MIN_SCORE = None, float("inf"), {}
        for id, cost in self.idx.items():
            #   (1) Normalization & Weight
            if self.Frechet_range[0]   != 0: cost["Frechet"]     *= self.weight["Frechet"]     / self.Frechet_range[0]
            if self.Dest_range[0]      != 0: cost["Destination"] *= self.weight["Destination"] / self.Dest_range[0]
            if self.Control_range[0]   != 0: cost["Controls"]    *= self.weight["Controls"]    / self.Control_range[0]
            if self.Obst_range[0]      != 0: cost["Obstacles"]   *= self.weight["Obstacles"]   / self.Obst_range[0]
            if self.Collision_range[0] != 0: cost["Collision"]   *= self.weight["Collision"]   / self.Collision_range[0]
            if self.Dist_range[0]      != 0: cost["Disturbance"] *= self.weight["Disturbance"] / self.Dist_range[0]
            
            #   (2) Find MIN & Avoid cost-areas
            cost["Total"] = sum(cost.values())
            if not AVOID: cost["Total"] -= cost["Disturbance"]
            if cost["Total"] < MIN_COST: MIN_ID, MIN_COST, MIN_SCORE = id, cost["Total"], cost
            if not AVOID: cost["Total"] += cost["Disturbance"]
        return MIN_ID[0], MIN_ID[1], self.idx[MIN_ID]["Total"], MIN_SCORE, self.source[MIN_ID]
    def check(self) -> bool:
        BLIND, SAFE = True, len(self.idx)
        for cost in self.idx.values():
            if cost["Collision"] == 0: BLIND = False
            if cost["Disturbance"] != 0: SAFE -= 1
        return BLIND, SAFE/len(self.idx) > 0.1
    
    def show(self, ID=False) -> None:
        MIN  = utility.Colors.Pattern(text=utility.Colors.TLCYAN)
        CLR  = utility.Colors.Pattern(text=utility.Colors.TDGREY)
        COST = utility.Colors.Pattern(text=utility.Colors.TYELLOW)
        EDGE = utility.Colors.Pattern(text=utility.Colors.TRED)
        END  = utility.Colors.END
        print("\n")
        print(f"\t{'ID':<15}{'Frechet':<15}{'Dest':<15}{'Obst':<15}{'Ctrl':<15}{'Collision':<15}{'Disturbance':<15}{'Total':<15}")
        print("\t" + "="*125)
        for idx, cost in sorted(self.idx.items(), key=lambda x: self.source[x[0]]):
            if ID and idx[1] == ID: print(f"\t{MIN}{str(idx):<15}{END}{cost['Frechet']:<15.10f}{cost['Destination']:<15.10f}{cost['Obstacles']:<15.10f}", end="")
            elif self.source[idx] == "OnBD": print(f"\t{CLR}{str(idx):<15}{END}{cost['Frechet']:<15.10f}{cost['Destination']:<15.10f}{cost['Obstacles']:<15.10f}", end="")
            else: print(f"\t{EDGE}{str(idx):<15}{END}{cost['Frechet']:<15.10f}{cost['Destination']:<15.10f}{cost['Obstacles']:<15.10f}", end="")
            print(f"{cost['Controls']:<15.10f}{cost['Collision']:<15.10f}{cost['Disturbance']:<15.10f}{COST}{cost['Total']:<10.5f} ({self.source[idx]}){END}")
        print('\n')

# ============================================================================================================================= #

def plot_iter(iter=None, agent=None, blind: list=[], TRIM:  bool=False,
                                                     FOCUS: bool=False,
                                                     SHOW:  bool=False,
                                                     SAVE:  bool=False,
                                                     AREA:  bool=False):
    PLT.close('all')

    #   (1) MAP
    fig = agent.map.get_plot(title=f"Iteration: {iter}", origin=agent.args.origin, dest=agent.args.dest)
    fig.subplots_adjust(top=0.99, bottom=0.16, left=0.01, right=0.99)

    #   | Agent
    PLT.scatter(agent.cur[0], agent.cur[1], color='#1434A4', marker='x', label="Agent")
    zipped = list(zip(*agent.prev))
    PLT.plot(zipped[0], zipped[1], color='#1434A4', alpha=0.5, marker='o', linewidth=1, markersize=3, label="Agent Path", zorder=999)
    
    #   | Blind
    if len(blind) != 0:
        blind = list(zip(*blind))
        PLT.plot(blind[0], blind[1], color='#FFEA00', marker='p', linewidth=0, markersize=3, zorder=999, label="Blind Spot")
    
    #   (2) PATH
    PLT.plot(agent.ref[0], agent.ref[1], c='#7393B3', alpha=0.1, marker='o', linewidth=3, markersize=3, label="Reference Path")
    
    #   (3) TRAJ
    MIN_x = MIN_y = float("inf")
    MAX_x = MAX_y = float("-inf")
    for idx, sampling in zip(agent.idx, agent.pred):
        MIN_x, MIN_y = min(MIN_x, torch.min(sampling[:,0])), min(MIN_y, torch.min(sampling[:,1]))
        MAX_x, MAX_y = max(MAX_x, torch.max(sampling[:,0])), max(MAX_y, torch.max(sampling[:,1]))
        if FOCUS:
            PLT.plot(sampling[:,0], sampling[:,1], c="#F28C28", alpha=0.3, linewidth=2, label="Trajectory", zorder=999)
            PLT.annotate('{}'.format(idx),
                xy=(sampling[-1][0], sampling[-1][1]), xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points", ha='center', va='bottom', c="#F28C28", zorder=999)
    
    if agent.args.nodes != 0:
        #   (4) NODES
        nodes = NP.array([agent.map.edge_nodes[key]["location"] for key in agent.map.edge_nodes])
        for idx, node in enumerate(nodes):
            if agent.args.nodes: PLT.scatter(node[0], node[1], marker=f'H', c='red', alpha=0.2, label=f"Server")
            else: PLT.scatter(node[0], node[1], marker=f'H', c='grey', alpha=0.2, label=f"Server")
    
        #   | Connections
        nodes = set()
        for server in agent.connections.values(): nodes |= server
        for node in nodes: PLT.plot([agent.map.edge_nodes[node]["location"][0], agent.cur[0]], [agent.map.edge_nodes[node]["location"][1], agent.cur[1]],
                                    ls="--", c="red", alpha=0.5, label=f"Connection")

    if AREA: 
        #   | Coverage
        info = agent.map.edge_nodes[agent.EDGE_ID]
        PLT.plot(info["region"].exterior.xy[0], info["region"].exterior.xy[1], zorder=0, c='#C41E3A', linewidth=2, label="Area")
        
        #   | Filtered Obstacles
        obst = info["obstacles"]
        obst_collection = PatchCollection([Rectangle(xy=obst[i,:2], width=obst[i,2], height=obst[i,3]) for i in range(len(obst))], facecolor='black', edgecolor='white', hatch="/")
        PLT.gca().add_collection(obst_collection)

    #   (*) TRIM
    if TRIM:
        PLT.xlim([min(agent.ref[0])*0.95, max(agent.ref[0])*1.05])
        PLT.ylim([min(agent.ref[1])*0.95, max(agent.ref[1])*1.05])
        # PLT.xlim([0.5*0.95, 0.8*1.05])
        # PLT.ylim([0.5*0.95, 0.8*1.05])

    #   (*) FOCUS
    if FOCUS:
        PLT.xlim([MIN_x*0.97, MAX_x*1.03])
        PLT.ylim([MIN_y*0.97, MAX_y*1.03])

    #   (*) LEGEND
    handles, labels = PLT.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    PLT.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.005))

    #   (*) SHOW
    PLT.gca().axes.xaxis.set_ticks([])
    PLT.gca().axes.yaxis.set_ticks([])
    if SHOW:
        if SAVE: fig.savefig(f"{DATE.datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf")
        fig.show()
        input()
    return fig

def algo(agent=None):
    iter       = agent.args.idx
    points     = agent.points
    ref        = agent.ref
    agent.cur  = agent.origin
    agent.prev = []

    #   (Experiment Scenario)
    CASE = "0-OnBD"
    if   agent.args.edge != 0 and agent.args.nodes != 0:
        utility.log(option="syst", msg=f"-edges {agent.args.edge} | -nodes {agent.args.nodes}", color=True, pattern=utility.Colors.Pattern(text=utility.Colors.TRED))
        SYS.exit()
    if   agent.args.edge != 0 and agent.args.nodes == 0: CASE = "1-EDGE"
    elif agent.args.edge == 0 and agent.args.nodes != 0: CASE = "3-SERVER"

    func_within = lambda x, y: y[0] > x[0][0] and y[0] < x[0][0]+x[1] and y[1] > x[0][1] and y[1] < x[0][1]+x[2]
    weight_NOR  = {"Frechet": 0.47, "Destination": 0.00, "Obstacles": 0.43, "Controls": 0.10, "Collision": 5, "Disturbance": 4}
    weight_LOS  = {"Frechet": 0.00, "Destination": 0.70, "Obstacles": 0.20, "Controls": 0.10, "Collision": 5, "Disturbance": 4}
    CP          = [RD.choice([5, 10, 20, 40]) for _ in range(agent.args.nodes)]

    #   (Collected Data)
    COST    = 0.0
    DIST    = 0.0
    ASSIST  = [0, 0]
    DATA    = {}
    BLIND   = []

    """  E-MPC  """
    for _ in utility.ProgressBar(range(iter, points), f"{' ' * 39}Iteration: "):
        """  Re-calibrate Ref Path  """
        if agent.args.update and ((_+ 1) % 100 == 0):
            if agent.args.plot:
                PLT.close("all")
                plot_iter(iter=iter, agent=agent, blind=BLIND, TRIM=True, FOCUS=False, SHOW=True, SAVE=False)
                agent.show(iter=iter, cost=COST, dist=DIST, assist=ASSIST, blind=BLIND, traj=TRAJ_ID, cp=CP)
            
            prev = Copy(agent.path)
            agent.path.origin = agent.cur[:-1]
            agent.path.FIND = False
            agent.path.run(restrict=True, debug=False, PLOT=False)
            if prev.cost/prev.points > agent.path.cost/agent.path.points:
                agent.points, agent.ref = agent.path.points, agent.path.ref_div
                iter, points, ref = 0, agent.points, agent.ref
            else:
                agent.path = prev

            if agent.args.plot:
                agent.plot_ref(ref=agent.path.ref, TRIM=True)
                plot_iter(iter=iter, agent=agent, blind=BLIND, TRIM=True, FOCUS=False, SHOW=True, SAVE=False)

        """  Status  """
        FAIL = Map.point_has_collision(agent.cur[:-1], agent.map.obst)
        if FAIL: print(); break
        CLEAR = not Map.edge_has_collision(NP.array(agent.cur[:-1]), NP.array(agent.dest), agent.map.obst)
        STUCK = agent.args.sensing and (func_within(agent.map.cost_area[0], agent.cur) or func_within(agent.map.cost_area[1], agent.cur))
        AVOID = (NP.linalg.norm(NP.array(agent.dest) - NP.array(agent.cur[:-1])) > 0.05) and (agent.args.edge or agent.args.nodes)  # TODO
        
        """  Pre-proc  """
        costs = Cost(weight=weight_NOR) if not CLEAR else Cost(weight=weight_LOS)
        table = {}
        end = min(points, iter+agent.args.H)
        ref_eval = [ref[0][iter:end], ref[1][iter:end]]
        steps = min(end-iter, agent.args.H)

        """
        # ===============================================================================
        #   [MPC]   Server Connection                                                   =
        # ===============================================================================
        """
        #   (Filtering obstacles)
        agent.EDGE_ID, agent.connections, conn_dist, server_comp = agent.connect(cur=agent.cur, CP=CP)
        
        """
        # ===============================================================================
        #   [MPC]     Onboard Computing                                                 =
        # ===============================================================================
        """
        OnBD_IDX, OnBD_TRAJ, OnBD_PRED = agent.exec(cur=agent.cur)
        for i in range(len(OnBD_IDX)): costs.add(id=(i, int(OnBD_IDX[i])), data=agent.eval(steps=steps, ref=ref_eval, pred=OnBD_PRED[i]), src="OnBD")
        table["OnBD"] = OnBD_PRED

        if   CASE == "1-EDGE":
            """
            # ===============================================================================
            #   [E-MPC]   Increased Computational Capacity                                  =
            # ===============================================================================
            """
            edge = Agent(args=agent.args, num=agent.args.edge, map_obj=agent.map, path=agent.path, motion=agent.motion, ctrls=agent.ctrls, dtype=agent.dtype)
            edge.EDGE_ID = agent.EDGE_ID
            EDGE_IDX, EDGE_TRAJ, EDGE_PRED = edge.exec(cur=agent.cur)
            for i in range(len(EDGE_IDX)): costs.add(id=(i, int(EDGE_IDX[i])), data=agent.eval(steps=steps, ref=ref_eval, pred=EDGE_PRED[i]), src="EDGE")
            table["EDGE"] = EDGE_PRED

        elif CASE == "3-SERVER":
            """
            # ===============================================================================
            #   [E-MPC]   Multiple Servers                                                  =
            # ===============================================================================
            """
            for idx, channel in agent.connections.items():
                for server in channel:
                    #   (server not available)
                    if NP.random.random_sample() > float(agent.args.avail):
                        if agent.args.debug: utility.log(option="empc", msg=f"Edge Server {server} is offline at iteration {iter}")
                        continue

                    edge = Agent(args=agent.args, num=server_comp[server], map_obj=agent.map, path=agent.path, motion=agent.motion, ctrls=agent.ctrls, dtype=agent.dtype)
                    edge.EDGE_ID = agent.EDGE_ID
                    EDGE_IDX, EDGE_TRAJ, EDGE_PRED = edge.exec(cur=agent.cur)
                    for i in range(len(EDGE_IDX)): costs.add(id=(i, int(EDGE_IDX[i])), data=agent.eval(steps=steps, ref=ref_eval, pred=EDGE_PRED[i]), src=str(server))
                    table[str(server)] = EDGE_PRED
        
        """
        # ===============================================================================
        #   [Cost]     Evaluation                                                       =
        # ===============================================================================
        """
        CHECK = costs.check()   # BLIND | SAFE (disturbance ratio)
        if CHECK[0]: BLIND.append(agent.cur)
        PRED_ID, TRAJ_ID, MIN_COST, SCORE, SOURCE = costs.eval(AVOID=AVOID and CHECK[1])

        # =====

        """  Move  """
        iter += 1
        agent.prev.append(agent.cur)
        agent.cur = table[str(SOURCE)][PRED_ID][1].tolist()
        if STUCK:
            """  Mud Area  """
            # agent.cur[0], agent.cur[1] = (agent.cur[0]+agent.prev[-1][0])/2, (agent.cur[1]+agent.prev[-1][1])/2

            """  Icy Road  """
            # agent.cur[0], agent.cur[1] = agent.cur[0]+(agent.cur[0]-agent.prev[-1][0])/2, agent.cur[1]+(agent.cur[1]-agent.prev[-1][1])/2
        
        """  Collect  """
        COST += MIN_COST
        if agent.args.nodes and server_comp: DIST += conn_dist / len(server_comp)
        ASSIST[0] += len(costs.idx) - agent.num
        ASSIST[1] += SOURCE != "OnBD"
        if agent.args.save: DATA[(_, tuple(agent.prev[-1]))] = TRAJ_ID

        """  Check  """
        GAP = NP.linalg.norm(NP.array(agent.dest)[:-1] - NP.array(agent.cur)[:-1])
        END = (GAP <= agent.args.dist) and CLEAR
        if END: print(); break

        """  Debug  """
        # ---
        if (iter + 1) % agent.args.epoch == 0: 
            if agent.args.debug: costs.show(ID=TRAJ_ID); agent.show(iter=_, cost=COST, dist=DIST, assist=ASSIST, blind=BLIND, traj=TRAJ_ID, cp=server_comp)
            if agent.args.debug and not agent.args.plot: input()
            if agent.args.plot: plot_iter(iter=iter, agent=agent, TRIM=True, FOCUS=False, SHOW=True, SAVE=False)


    # ==========


    if END:
        print()
        pattern = utility.Colors.Pattern(feature=utility.Colors.BOLD, text=utility.Colors.TLCYAN)
        utility.log(option="empc", msg=f"Reach destination around {GAP:.10f} at iteration {iter}", color=True, pattern=pattern)
    elif FAIL:
        print()
        pattern = utility.Colors.Pattern(feature=utility.Colors.BOLD, text=utility.Colors.TRED)
        utility.log(option="empc", msg=f"Collide with obstacles at iteration {iter}", color=True, pattern=pattern)
    else:
        pattern = utility.Colors.Pattern(feature=utility.Colors.BOLD, text=utility.Colors.TRED)
        utility.log(option="empc", msg=f"Fail to reach destination", color=True, pattern=pattern)


    """
    # ===============================================================================
    #   [Save]     Results                                                          =
    # ===============================================================================
    """
    if agent.args.sensing and agent.args.edge != 0:     CASE = "2-SENSE"
    if agent.args.avail != 1:                           CASE = "4-AVAIL"
    if agent.args.latency != 0:                         CASE = "5-LATENCY"
    COST += len(BLIND) * 5
    RESULT = {
        "case": CASE,
        "map": agent.map.random_seed,
        "iter": _,
        "status": str(END),
        "cost": COST,
        "blind": len(BLIND),
        "config": {
            "update":   agent.args.update,      # DONE
            "onboard":  agent.args.num,         # DONE
            "edge":     agent.args.edge,        # DONE
            "sensing":  agent.args.sensing,     # DONE; criteria
            "nodes":    agent.args.nodes,       # DONE
            "avail":    agent.args.avail,       # DONE
            "latency":  agent.args.latency,     # DONE; optimized
        },
    }
    RESULT["distance"], RESULT["server"] = DIST, dict(enumerate(CP))
    RESULT["assist"] = ASSIST
    INFO = JSON.dumps(RESULT, indent=4)
    RESULT["args"] = agent.args.__dict__
    if agent.args.save: RESULT["path"] = DATA
    FILE = f"_data/map-{agent.map.random_seed}-case-{CASE}-{'O' if END else 'X'}-cost-{COST:.5f}-date-{DATE.datetime.now().strftime('%Y%m%d-%H%M%S')}.data"
    
    """  Plot  |  Debug  |  Save  |  Show  """
    agent.show(iter=_, cost=COST, dist=DIST, assist=ASSIST, blind=BLIND, traj=TRAJ_ID, cp=server_comp)
    RESULT["figure"] = plot_iter(iter=iter, agent=agent, blind=BLIND, TRIM=True, FOCUS=False, SHOW=False, SAVE=False)
    if agent.args.save: torch.save(RESULT, FILE)
    utility.log(option="empc", msg=f"Save Result: {FILE}\n", color=True)
    print(INFO)

    return RESULT