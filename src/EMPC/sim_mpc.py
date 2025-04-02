# my attemp at model predictive control
from libgen import MPdict
from motion import Motion
import torch

class utility:
  
  def __init__(self):
    self.N = 1000
    self.dist = 2
    self.wheel = 0.5
    self.H = 30
    self.segments = 3
    self.branch = 10
    self.speed = 1.0
    self.range = [-0.68, 0.68]

def main():
  # N, dist, wheel, H
  args = utility()
  motion = Motion(args=args, dtype=torch.FloatTensor)
  library = MPdict(args=args, motion=motion,  dtype=torch.FloatTensor)
  ctrls, fig_ctrl = library.get()
  fig_ctrl.show(); input()
  
    
if __name__ == "__main__":
  main()
      