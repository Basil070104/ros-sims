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

Project : Python3 - Utility functions
Version : v2.0
"""


import os                       as OS
import matplotlib.pyplot        as PLT
import sys                      as SYS
import re                       as RE
import json                     as JSON

import configparser             as ConfigParser
import datetime                 as DateTime
import subprocess               as SubProc


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self): return JSON.dumps(self.__dict__, indent=4, separators=(',', ': '))
    def get(self): return [f"{' '*39 if idx > 0 else ''}-{k:<15}  {v}\n" for idx, (k, v) in enumerate(self.__dict__.items())]
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

class Colors:
    END   = "\033[0m"

    # Feature
    NORM  = "0"
    BOLD  = "1"
    TRANS = "3"
    LINE  = "4"
    REV   = "7"
    DEL   = "9"
    NONE  = "8"

    # Text
    TWHITE   = "0"
    TBLACK   = "30"
    TRED     = "31"
    TGREEN   = "32"
    TYELLOW  = "33"
    TBLUE    = "34"
    TPURPLE  = "35"
    TCYAN    = "36"
    TGREY    = "37"
    TDGREY   = "90"
    TLRED    = "91"
    TLGREEN  = "92"
    TLYELLOW = "93"
    TLBLUE   = "94"
    TLPURPLE = "95"
    TLCYAN   = "96"

    # Background
    BNONE    = "10"
    BBLACK   = "40"
    BRED     = "41"
    BGREEN   = "42"
    BYELLOW  = "43"
    BBLUE    = "44"
    BPURPLE  = "45"
    BCYAN    = "46"
    BLGREY   = "47"

    def Pattern(feature="0", text="0", background="10"):
        """
        Return the color pattern.

        args:
            @feature:       str
            @text:          str
            @background:    str
        """

        return "\033[{0};{1};{2}m".format(feature, text, background)

def getTime() -> str:
    """
    Get current time in format (%Y-%m-%d %H:%M:%S:%f).
    """

    return f"{DateTime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]}"

def convertTime(time: str) -> object:
    """
    Convert time in string format to datetime ojbect.

    args:
        @time           str
    """

    return DateTime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S:%f")

def log(option: str, msg: str, color: bool=False, pattern: str=None) -> str:
    """
    Systen log

    args:
        @option:        str     [syst]
        @msg:           str
        @client:        str
        @color:         bool
        @pattern:       str
    """

    alert = None
    if option == "syst":
        default = Colors.Pattern(Colors.NORM, Colors.TYELLOW, Colors.BNONE)
        alert = f"{default} [{option[0].upper()}{option[1:]}]    {Colors.END}"
    
    if option == "path":
        default = Colors.Pattern(Colors.NORM, Colors.TWHITE, Colors.BNONE)
        alert = f"{default} [{option[0].upper()}{option[1:]}]    {Colors.END}"
    
    if option == "empc" or option == "edge":
        default = Colors.Pattern(Colors.NORM, Colors.TLCYAN, Colors.BNONE)
        alert = f"{default} [{option.upper()}]    {Colors.END}"

    TIME = Colors.Pattern(Colors.NORM, Colors.TBLUE, Colors.BNONE)
    MSG  = pattern if pattern != None else default

    time = getTime()
    print(f"{TIME + time + Colors.END:<40} {alert} {MSG + msg + Colors.END if color else msg}")

    return time

def ProgressBar(it, prefix="", size=60, out=SYS.stdout):
    count = len(it)
    pattern = Colors.Pattern(text=Colors.TGREEN)
    end = Colors.END
    
    def show(j):
        x = int(size * j / count)
        # print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
        print(f"{pattern}{prefix}[{u'='*x}{('.'*(size-x))}] {j}/{count}{end}", end='\r', file=out, flush=True)
    
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)