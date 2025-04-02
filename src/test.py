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
Version : v3.1
"""

# ----
path = './default.config'
map_id = 160
num_sampling = 10
weight_NOR = {"Frechet": 0.47, "Destination": 0.00, "Obstacles": 0.43, "Controls": 0.10, "Collision": 5, "Disturbance": 4}
weight_LOS = {"Frechet": 0.00, "Destination": 0.70, "Obstacles": 0.20, "Controls": 0.10, "Collision": 5, "Disturbance": 4}
# ----
import datetime
import json
import matplotlib.pyplot        as plt
from matplotlib.patches         import Rectangle
from matplotlib.collections     import PatchCollection
import numpy                    as np
import similaritymeasures
from shapely.geometry           import Point
import sys
import torch
# ----
import EMPC.utility             as utility
import EMPC.map                 as Map
from   EMPC.path                import Path
from   EMPC.motion              import Motion
import EMPC.libgen              as Lib
# ----
class Agent:

    """
    Cusotom autonomus vehicle (AV) agent
    """

    NCTRL = 2
    NPOS  = 3

    def __init__(self, args, env, path, motion, ctrls, dtype) -> None:
        self.args   = args
        self.map    = env
        self.origin = self.args.origin
        self.dest   = self.args.dest[:-1]
        self.path   = path
        self.points = self.path.points
        self.ref    = self.path.ref_div
        self.motion = motion
        self.ctrls  = ctrls
        self.dtype  = dtype
        utility.log(option="empc", msg=f"Agent Initialization")

        self.origin, traj = self.calibrate()
        utility.log(option="empc", msg=f"Place starting point at {self.origin}")

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
        if iterIdx is None: iterIdx = self.args.idx
        ref = [self.ref[0][iterIdx:min(self.points,iterIdx+steps)], self.ref[1][iterIdx:min(self.points,iterIdx+steps)]]

        for angle in range(-500, 500):                              # Vision range
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
                path = np.array(path)[:, :min(self.points-iterIdx,steps)]
                xy = [np.array(path[0]), np.array(path[1])]
                cost_frechet = similaritymeasures.frechet_dist(xy, ref)

                if cost_frechet < MIN_cost:
                    origin = [origin[0],origin[1],angle/100.0]
                    MIN_cost = cost_frechet
                    traj = pred[idx]

            self.motion.set(self.args.N)
        
        return origin, self.dtype([list(zip(*traj))])

    @staticmethod
    def sample(ctrls, num):
        idx = torch.randperm(ctrls.shape[0])
        traj = ctrls[idx].view(ctrls.size())[:num]
        idx = idx[:num]
        return idx, traj

    @staticmethod
    def predict(motion, poses, ctrls, horizon):
        for t in range(1, horizon):
            cur_pose = poses[:, t - 1]
            cur_ctrl = ctrls[:, t - 1]
            poses[:, t] = motion.apply(cur_pose, cur_ctrl)
        return poses

    def mpc(self, cur):
        """
        Sampling-based Model Predicitive Control (MPC)
        """

        # (input)
        ctrls = self.ctrls
        motion = self.motion
        num = self.num
        H = self.args.H

        # Sample
        idx, traj = self.sample(ctrls=ctrls, num=num)

        # Predict
        n_path = len(traj)
        pred = self.dtype(n_path, H, self.NPOS).zero_()
        pred[:] = self.dtype(cur)
        self.motion.set(size=num)
        self.pred = self.predict(motion=motion, poses=pred, ctrls=traj, horizon=H)

        self.idx, self.traj = idx, traj
        return self.idx, self.traj, self.pred
    
    def eval(self, steps, ref, pred, EDGE_ID):
        """
        Cost Evaluation
            - Frechet distance
            - Distance from destination
            - Magnitude of steering control
            - Distance from obstacles
            - Collision detection
            - Extra disturbance
        """

        pred = np.array(pred)[:steps, :]                # [[x, y, theta], ...]
        x, y, steering = list(zip(*pred))               # [x1, x2, ...], [theta1, theta2, ...]
        xy, cords = [x, y], list(zip(x, y))             # [x, y], [[x, y], ...]
        func_within = lambda x, y: y[0] > x[0][0] and y[0] < x[0][0]+x[1] and y[1] > x[0][1] and y[1] < x[0][1]+x[2]

        costs = {"Frechet": 0.0, "Destination": 0.0, "Obstacles": 0.0, "Controls": 0.0, "Collision": 0, "Disturbance": 0.0}
        costs["Frechet"] = similaritymeasures.frechet_dist(xy, ref)
        costs["Destination"] = np.linalg.norm(np.array(self.dest) - np.array(cords[-1]))
        costs["Controls"] = float(np.sum(np.abs(np.diff(pred[:,-1]))))
        obst_dist = self.map.get_obst_dist(cord=cords[len(cords) // 2], FILTER=EDGE_ID)
        costs["Obstacles"] = 1 / obst_dist if obst_dist else 0.0
        costs["Collision"] = sum([1/i if Map.point_has_collision(cords[i], self.map.obst) else 0 for i in range(1, len(cords))])
        if self.map.cost_area: costs["Disturbance"] = int(sum([int(func_within(self.map.cost_area[0], cur) or func_within(self.map.cost_area[1], cur)) for cur in cords]))

        return costs

    def show(self, iter=None, cost=None, blind=None, traj=None):
        COLOR = utility.Colors.Pattern(text=utility.Colors.TYELLOW)
        END = utility.Colors.END
        print(f"\t{'X':<25}{'Y':<25}{'Angle':<25}{'# Trajectory ID'}")
        print("\t" + "="*125)
        print(f"\t{self.cur[0]:<25.15f}{self.cur[1]:<25.15f}{self.cur[2]:<25.15f}{COLOR}{traj}{END}")
        print()
        print(f"\t{'Iteration':<25}{'Total Cost':<25}{'Blind Spots'}")
        print("\t" + "="*125)
        print(f"\t{COLOR}{iter:<25}{cost:<25.10f}{len(blind):<25}{END}")
        print("\n")

    @staticmethod
    def plot(iter=None, agent=None, blind: list=[], EDGE_ID=None, AREA: bool=False, TRIM: bool=False, FOCUS: bool=False, SAVE: bool=False):
        plt.close('all')

        # Map
        fig = agent.map.get_plot(title=f"Iteration: {iter}", origin=agent.args.origin, dest=agent.args.dest)
        fig.subplots_adjust(top=0.99, bottom=0.16, left=0.01, right=0.99)

        # Agent
        plt.scatter(agent.cur[0], agent.cur[1], color='#1434A4', marker='x', label="Agent")
        zipped = list(zip(*agent.prev))
        plt.plot(zipped[0], zipped[1], color='#1434A4', alpha=0.5, marker='o', linewidth=1, markersize=3, label="Agent Path", zorder=999)
        
        # Danger zone
        if len(blind) != 0:
            blind = list(zip(*blind))
            plt.plot(blind[0], blind[1], color='#FFEA00', marker='p', linewidth=0, markersize=3, zorder=999, label="Blind Spot")
        
        # Global path
        plt.plot(agent.ref[0], agent.ref[1], c='#7393B3', alpha=0.1, marker='o', linewidth=3, markersize=3, label="Reference Path")
        
        # Trajectory
        MIN_x = MIN_y = float("inf")
        MAX_x = MAX_y = float("-inf")
        for idx, sampling in zip(agent.idx, agent.pred):
            MIN_x, MIN_y = min(MIN_x, torch.min(sampling[:,0])), min(MIN_y, torch.min(sampling[:,1]))
            MAX_x, MAX_y = max(MAX_x, torch.max(sampling[:,0])), max(MAX_y, torch.max(sampling[:,1]))
            if FOCUS:
                plt.plot(sampling[:,0], sampling[:,1], c="#F28C28", alpha=0.3, linewidth=2, label="Trajectory", zorder=999)
                plt.annotate('{}'.format(idx),
                    xy=(sampling[-1][0], sampling[-1][1]), xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom', c="#F28C28", zorder=999)
        
        # Server nodes
        nodes = np.array([agent.map.edge_nodes[key]["location"] for key in agent.map.edge_nodes])
        for idx, node in enumerate(nodes):
            if agent.args.nodes: plt.scatter(node[0], node[1], marker=f'H', c='red', alpha=0.2, label=f"Server")
            else: plt.scatter(node[0], node[1], marker=f'H', c='grey', alpha=0.2, label=f"Server")

        if AREA:
            # Coverage
            if EDGE_ID:
                info = agent.map.edge_nodes[EDGE_ID]
                plt.plot(info["region"].exterior.xy[0], info["region"].exterior.xy[1], zorder=0, c='black', linewidth=2, label="Area")
            
            # Filtered Obstacles
            obst = info["obstacles"]
            obst_collection = PatchCollection([Rectangle(xy=obst[i,:2], width=obst[i,2], height=obst[i,3]) for i in range(len(obst))], facecolor='brown', edgecolor='white')
            plt.gca().add_collection(obst_collection)

        # (trimmed view)
        if TRIM:
            plt.xlim([min(agent.ref[0])*0.95, max(agent.ref[0])*1.05])
            plt.ylim([min(agent.ref[1])*0.95, max(agent.ref[1])*1.05])

        # (focused view)
        if FOCUS:
            plt.xlim([MIN_x*0.97, MAX_x*1.03])
            plt.ylim([MIN_y*0.97, MAX_y*1.03])

        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.005))

        # show
        plt.gca().axes.xaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticks([])
        if SAVE: fig.savefig(f"./{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf")
        # fig.show()
        # input()
        
        return fig
# ----
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
# ----

# Configuration
try: config = json.load(open(path, 'r'))
except: sys.exit(f'Cannot find configuration: {path}.\n')
args = utility.Namespace(**config)
print(json.dumps(args.__dict__, indent=4))

# Map
args.map = map_id
map, fig_map = Map.init(args)

# (plot)
if args.plot:
    fig_map.show(); input()
    plt.close('all')

# Global path
path = Path(args, PATH=f"./pth/{map_id}.pth", samplings=20, wnd=7, map=map)
path.run(debug=False, PLOT=False)

# (plot)
if args.plot:
    path.plot(path.paths).show(); input()
    plt.close('all')

# Motion model
motion = Motion(args, dtype=torch.FloatTensor)

# Trajectory library
mpdict = Lib.MPdict(args, motion=motion, dtype=torch.FloatTensor)
ctrls, fig_ctrl = mpdict.get()

# (plot)
if args.plot:
    fig_ctrl.show(); input()
    plt.close('all')

# Autonomous Vehicle (AV) agent
agent = Agent(args=args, env=map, path=path, motion=motion, ctrls=ctrls, dtype=torch.FloatTensor)

# (variables)
iter = agent.args.idx
points = agent.points
ref = agent.ref
agent.cur = agent.origin
agent.prev = []
agent.num = num_sampling
func_within = lambda x, y: y[0] > x[0][0] and y[0] < x[0][0]+x[1] and y[1] > x[0][1] and y[1] < x[0][1]+x[2]

"""  Data  """
cost = 0.0      # Performance
table = {}      # Cost evaluation
danger = []     # Blind spots
data = {}       # Driving path

# Navigation
END = False
FAIL = False
for _ in utility.ProgressBar(range(iter, points), f"{' ' * 39}Iteration: "):
    
    """  Status  """
    FAIL = Map.point_has_collision(agent.cur[:-1], agent.map.obst)
    if FAIL: print(); break
    CLEAR = not Map.edge_has_collision(np.array(agent.cur[:-1]), np.array(agent.dest), agent.map.obst)
    STUCK = agent.args.sensing and (func_within(agent.map.cost_area[0], agent.cur) or func_within(agent.map.cost_area[1], agent.cur))
    AVOID = (np.linalg.norm(np.array(agent.dest) - np.array(agent.cur[:-1])) > 0.05) and (agent.args.edge or agent.args.nodes)
    NODE = np.argmax([info["region"].contains(Point(agent.cur[0], agent.cur[1])) for info in agent.map.edge_nodes.values()])

    """  Cost Function """
    costs = Cost(weight=weight_NOR) if not CLEAR else Cost(weight=weight_LOS)
    end = min(points, iter+agent.args.H)
    ref_eval = [ref[0][iter:end], ref[1][iter:end]]
    steps = min(end-iter, agent.args.H)

    src = 'onboard'
    idx, traj, pred = agent.mpc(cur=agent.cur)
    for i in range(len(idx)): costs.add(id=(i, int(idx[i])), data=agent.eval(steps=steps, ref=ref_eval, pred=pred[i], EDGE_ID=NODE), src=src)
    table[src] = pred

    """  Evaluation  """
    BLIND, SAFE = costs.check()
    if BLIND: danger.append(agent.cur)
    pred_id, traj_id, min_cost, score, src = costs.eval(AVOID=AVOID and SAFE)

    """  Move  """
    iter += 1
    agent.prev.append(agent.cur)
    agent.cur = table[str(src)][pred_id][1].tolist()
    cost += min_cost
    if agent.args.save: data[(_, tuple(agent.prev[-1]))] = traj_id

    """  Check  """
    GAP = np.linalg.norm(np.array(agent.dest)[:-1] - np.array(agent.cur)[:-1])
    END = (GAP <= agent.args.dist) and CLEAR
    if END: print(); break

    # (debug, plot)
    if agent.args.debug:
        costs.show(ID=traj_id)
        agent.show(iter=_, cost=cost, blind=danger, traj=traj_id)
        agent.plot(iter=iter, agent=agent, blind=danger, EDGE_ID=NODE, TRIM=True, FOCUS=False, SAVE=False)

# Output
if END:
    msg=f"Reach destination around {GAP:.10f} at iteration {iter}"
    pattern=utility.Colors.Pattern(feature=utility.Colors.BOLD, text=utility.Colors.TLCYAN)
elif FAIL:
    msg=f"Collide with obstacles at iteration {iter}"
    pattern = utility.Colors.Pattern(feature=utility.Colors.BOLD, text=utility.Colors.TRED)
else:
    msg=f"Fail to reach destination"
    pattern = utility.Colors.Pattern(feature=utility.Colors.BOLD, text=utility.Colors.TRED)
print(); utility.log(option="empc", msg=msg, color=True, pattern=pattern)
RESULT = {
    "map": agent.map.random_seed,
    "iter": _,
    "status": str(END),
    "comp":  agent.args.num,
    "cost": cost + len(danger) * 5,
    "blind": len(danger),
}
INFO = json.dumps(RESULT, indent=4)
FILE = f"./map-{agent.map.random_seed}-status-{'O' if END else 'X'}-cost-{cost:.5f}-date-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.data"
utility.log(option="empc", msg=f"Save Result: {FILE}", color=True)
print(INFO)
fig = agent.plot(iter=iter, agent=agent, blind=danger, EDGE_ID=NODE, AREA=False, TRIM=True, FOCUS=False, SAVE=False)
RESULT['figure'] = fig
torch.save(RESULT, FILE)
