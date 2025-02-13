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
Version : v2.0
"""


import numpy                    as NP
import os                       as OS
import matplotlib.pyplot        as PLT
from itertools                  import product
from scipy.spatial.distance     import directed_hausdorff
import torch

# from utility                    import Namespace
# from utility                    import log


class MPdict:
    """
    Motion Primitive / Trajectory Dictionary
    
    Each primitive (alpha) consists of a sequence of controls [u1, ..., uH]
    (H = look-ahead window / horizon)
    """

    NCTRL = 2
    NPOS  = 3
    DELTAS_EPSILON = 0.0001

    def __init__(self, args, motion, dtype):
        self.args = args
        self.motion = motion
        self.dtype = dtype
        self.init()

    def init(self):
        self.total     = self.args.N                            # 1000
        self.H         = self.args.H                            # 30
        self.segments  = self.args.segments                     # 3
        self.branch    = self.args.branch                       # 10
        self.N         = self.branch ** self.segments           # 10^3
        self.t_seg     = self.H // self.segments                # 10
        self.speed     = self.args.speed                        # 1.0
        self.range     = self.args.range                        # -0.68, 0.68
        self.cfg_ID    = "{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}".format(self.total,self.H, self.segments, self.branch, self.range[1], self.args.dist, self.speed, self.args.wheel)

        min_range, max_range = self.range
        step_size      = (max_range - min_range) / (self.branch - 1)
        deltas         = torch.arange(min_range, max_range+self.DELTAS_EPSILON, step_size)

        self.PATH = "./lib/{0}.lib".format(self.cfg_ID)
        if OS.path.exists(self.PATH):
            self.ctrls = torch.load(self.PATH)
            # log(option="empc", msg="Load trajectory dictionary\t(FILE: {0})".format(self.PATH), color=True)
            print("Load trajectory dictionary\t(FILE: {0})".format(self.PATH))
            # if self.speed != 1.0: self.ctrls = torch.concat((self.ctrls, torch.load(self.PATH.replace(str(self.speed), "1.0"))))
            return self.ctrls
        
        # ---

        # log(option="empc", msg="Generate trajectory dictionary\t(FILE: {0})".format(self.PATH), color=True)
        print("Generate trajectory dictionary\t(FILE: {0})".format(self.PATH))

        cartesian_prod = product(*[deltas for i in range(self.segments)])
        tmp            = self.dtype(NP.array(list(cartesian_prod)))                                # torch.Size([N, segments])
        ms_deltas      = (tmp.view(-1, 1).repeat(1, self.t_seg).view(self.N, self.H))              # torch.Size([N, H])

        #   Add zero control (if not exists in ms_deltas)
        init_ctrl = self.dtype(self.H).zero_()
        idx = -1
        for i, c in enumerate(ms_deltas):
            if torch.equal(c, init_ctrl): idx = i
        if idx >= 0:
            ms_ctrls = self.dtype(self.N, self.H, self.NCTRL)                                       # torch.Size([branch ^ segments, H, CTRL])
            ms_ctrls[:, :, 0] = self.speed
            ms_ctrls[:, :, 1] = ms_deltas
        else:
            idx = 0
            ms_ctrls = self.dtype(self.N+1, self.H, self.NCTRL)
            ms_ctrls[:,  :, 0] = self.speed
            ms_ctrls[1:, :, 1] = ms_deltas
            ms_ctrls[0,  :, 1] = 0

        self.ms_poses = self.gen(ms_ctrls)
        self.ctrls = self.prune(idx, ms_ctrls, self.ms_poses)

        # log(option="empc", msg="Save trajectory dictionary\t(FILE: {})".format(self.PATH), color=True)
        print("Save trajectory dictionary\t(FILE: {})".format(self.PATH))
        torch.save(self.ctrls, self.PATH)

    def gen(self, ms_ctrls):
        total = self.motion.total
        self.motion.set(len(ms_ctrls))
        ms_poses = self.dtype(len(ms_ctrls), self.H, self.NPOS).zero_()
        for t in range(1, self.H):
            cur_pose = ms_poses[:, t - 1]
            cur_ctrl = ms_ctrls[:, t - 1]
            ms_poses[:, t] = self.motion.apply(cur_pose, cur_ctrl)
        self.motion.set(total)
        
        return ms_poses

    def prune(self, zero_idx, ms_ctrls, ms_poses):
        visited = {zero_idx: ms_poses[zero_idx]}
        dist_cache = {}
        hausdorff = lambda a, b: max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

        for _ in range(self.total - 1):
            max_i, max_dist = 0, 0
            for rollout in range(len(ms_ctrls)):
                if rollout in visited: continue

                min_dist = 10e10
                for idx, visited_rollout in visited.items():
                    if (idx, rollout) not in dist_cache:
                        d = hausdorff(visited_rollout[:, :2], ms_poses[rollout, :, :2])
                        dist_cache[(idx, rollout)] = d
                        dist_cache[(rollout, idx)] = d
                    min_dist = min(dist_cache[(idx, rollout)], min_dist)

                if min_dist > max_dist:
                    max_i, max_dist = rollout, min_dist

            visited[max_i] = ms_poses[max_i]

        assert len(visited) == self.total
        self.ctrls = self.dtype(self.total, self.H, self.NCTRL)
        self.ctrls.copy_(ms_ctrls[list(visited.keys())])

        return self.ctrls

    def plot(self):
        poses = self.dtype(len(self.ctrls), self.H, 3).zero_()
        poses[:] = self.dtype([0, 0, 0]) # [x, y, angle]
        for t in range(1, self.H):
            cur_pose = poses[:, t - 1]
            cur_ctrl = self.ctrls[:, t - 1]
            poses[:, t] = self.motion.apply(cur_pose, cur_ctrl)

        fig = PLT.figure("Trajectrory Library", figsize=(4, 5))
        fig.subplots_adjust(top=0.98, bottom=0.12, left=0.07, right=0.98)
        for line in poses[:, :, :-1]: PLT.plot(line[:, 0], line[:, 1], linestyle='--', marker='o')
        PLT.xticks(rotation=270)
        PLT.yticks(rotation=270)
        PLT.xlim()
        PLT.ylim()
        return fig

    def get(self, velocity=1):
        self.ctrls[:, :, 0] = velocity
        return self.ctrls, self.plot()