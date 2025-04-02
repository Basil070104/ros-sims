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


class Motion:
    """
    Motion model to simulate car moving with system dynamics
    """

    NCTRL = 2       # [speed, steering]
    NPOS  = 3       # [x, y, angle]
    EPSILON = 1e-5

    def __init__(self, args, dtype):
        self.args = args
        self.total = self.args.N
        self.dtype = dtype

        self.set(self.args.N)

    def set(self, size):
        """
        args:
            @size       int / Number of rollouts
        """
        self.total = size
        self.wheel = self.args.wheel

        dist = self.args.dist
        H = self.args.H
        self.dt = dist / H

        self.sin2beta   = self.dtype(size)
        self.deltaTheta = self.dtype(size)
        self.deltaX     = self.dtype(size)
        self.deltaY     = self.dtype(size)
        self.sin        = self.dtype(size)
        self.cos        = self.dtype(size)

    def apply(self, pose, ctrl):
        """
        args:
            @pose       [(K, NPOS) tensor]  / Current position
            @ctrl       [(K, NCTRL) tensor] / Control to apply to the current position
        
        return:
            [(K, NCTRL) tensor] / The next position given the current control
        """
        # print(pose.size(), (self.total, self.NPOS))
        # print(ctrl.size(), (self.total, self.NCTRL))
        assert pose.size() == (self.total, self.NPOS)
        assert ctrl.size() == (self.total, self.NCTRL)

        self.sin2beta.copy_(ctrl[:, 1]).tan_().mul_(0.5).atan_().mul_(2.0).sin_().add_(self.EPSILON)
        self.deltaTheta.copy_(ctrl[:, 0]).div_(self.wheel).mul_(self.sin2beta).mul_(self.dt)
        
        self.sin.copy_(pose[:, 2]).sin_()
        self.cos.copy_(pose[:, 2]).cos_()
        self.deltaX.copy_(pose[:, 2]).add_(self.deltaTheta).sin_().sub_(self.sin).mul_(self.wheel).div_(self.sin2beta)
        self.deltaY.copy_(pose[:, 2]).add_(self.deltaTheta).cos_().neg_().add_(self.cos).mul_(self.wheel).div_(self.sin2beta)

        nextpos = self.dtype(self.total, 3)
        nextpos.copy_(pose)
        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)
        nextpos[:, 2].add_(self.deltaTheta)

        return nextpos