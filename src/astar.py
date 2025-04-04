import node2
import cv2
import math
import matplotlib.pyplot as plt
import heapq
import numpy as np
import scipy.linalg as lu
from scipy.interpolate import CubicSpline
import sympy
import time

# Start Postion : X = 482, Y = 547
# End Position : X = 309, Y = 233
# Adjacency list with (id, cost)
# Starting Id : 438082

# Start and End Nodes have a dimension of 5 x 5
# x and y are the center of the nodes
def calculate_heuristic(x, y, end_x, end_y):
  # Diagnol Distance
  dx = abs(x - end_x)
  dy = abs(y - end_y)

  D = 1
  D2 = math.sqrt(2)
  h = D * (dx + dy) + (D2 - 2 * D) * min(dx,dy)
  return h

def draw_node(image, x, y):

  start_x = x - 2
  start_y = y - 2

  x = start_x
  y = start_y

  for i in range(0,5):
    for j in range(0,5):
      image[y][x] = 100
      x += 1
    x = start_x
    y += 1

def convert_to_nodes(image, height, width):

  nodes = list()
  end_x = 309
  end_y = 233

  id = 0
  for i in range(0, width):
    for j in range(0, height):
      if image[i][j] == 255:
        h = calculate_heuristic(j , i, end_x, end_y)
        nodes.append(node2.Node2(True, False, False, j, i, id, float('inf'), [], -1, h, float('inf')))
        id += 1
      elif image[i][j] == 0:
        h = calculate_heuristic(j , i, end_x, end_y)
        nodes.append(node2.Node2(True, False, True, j , i, id, float('inf'), [], -1, h, float('inf')))
        id += 1
      else:
        nodes.append(node2.Node2(False, False, False, j , i, id, float('inf'), [], -1, 0, float('inf')))
        id += 1

      if i == 547 and j == 482:
        nodes[id - 1].setStart(True)

      

  return nodes

def addEdge(adjlist, node_arr, id, cost):

  if id > len(node_arr) - 1:
    return

  if node_arr[id].isEdge() or not node_arr[id].active:
    return
  
  adjlist.append([id, cost])

def create_adjlist(nodes_arr):

  for gnodes in nodes_arr:
    if not gnodes.isEdge() and gnodes.active:
      temp_id = gnodes.id
      adjlist = []
      try :
        addEdge(adjlist, nodes_arr, temp_id - 1, 1)
        addEdge(adjlist, nodes_arr, temp_id + 1, 1)
        addEdge(adjlist, nodes_arr, temp_id - 800, 1)
        addEdge(adjlist, nodes_arr, temp_id + 800, 1)
        # addEdge(adjlist, nodes_arr, temp_id - 800 - 1, math.sqrt(2))
        # addEdge(adjlist, nodes_arr, temp_id - 800 + 1, math.sqrt(2))
        # addEdge(adjlist, nodes_arr, temp_id + 800 - 1, math.sqrt(2))
        # addEdge(adjlist, nodes_arr, temp_id + 800 + 1, math.sqrt(2))

        addEdge(adjlist, nodes_arr, temp_id - 800 - 1, 2)
        addEdge(adjlist, nodes_arr, temp_id - 800 + 1, 2)
        addEdge(adjlist, nodes_arr, temp_id + 800 - 1, 2)
        addEdge(adjlist, nodes_arr, temp_id + 800 + 1, 2)
      except:
        print(temp_id)
      
      gnodes.setAdjlist(adjlist)

def astar(image, nodes_arr):
  pq = []
  arr_len = len(nodes_arr)
  visited = [False] *  arr_len

  start_id = 0
  for gnode in nodes_arr:
    x , y = gnode.getCoordinates()
    if x == 482 and y == 547:
      start_id = gnode.getId()
      break

  # set the starting point in the pq
  nodes_arr[start_id].setDistance(0)
  nodes_arr[start_id].f = nodes_arr[start_id].distance + nodes_arr[start_id].h
  heapq.heappush(pq, nodes_arr[start_id])
  
  i = 0

  while len(pq) != 0:
  # for i in range(0,100):
    gnode = heapq.heappop(pq)
    id = gnode.id
    distance = gnode.distance
    f = gnode.f
    visited[id] = True
    adj_list = nodes_arr[id].getAdjlist()
    for adjlink in adj_list:
      if not visited[adjlink[0]]:
        curr_dis = nodes_arr[adjlink[0]].distance
        if distance + adjlink[1] < curr_dis:
          nodes_arr[adjlink[0]].distance = distance + adjlink[1]
          nodes_arr[adjlink[0]].pd = id
          nodes_arr[adjlink[0]].f = nodes_arr[adjlink[0]].distance + nodes_arr[adjlink[0]].h

          x, y = nodes_arr[adjlink[0]].getCoordinates()
          image[y][x] = 0
          heapq.heappush(pq, nodes_arr[adjlink[0]])

          if x == 309 and y == 233:
            return

def print_path(image, node_arr, end_id):

  solution = list()
  curr_id = end_id
  i = 0
  time = 0
  while curr_id != 438082:
    x, y = node_arr[curr_id].getCoordinates()
    image[y][x] = 200

    solution.insert(0, [x, y, 0])
    curr_id = node_arr[curr_id].pd 

    # For animating the search
    # if i > 50:
    #   plt.imshow(image)
    #   plt.pause(0.0001)
    #   i = 0
    # i+=1

  for coor in solution:
    coor[2] = time
    time+=0.1

  return solution

def generate_vector_trajectory(axis, solution):

  index = 0
  step_size = 10
  while index < len(solution) - step_size:
    # print("here")

    coordinate = solution[index]
    future_coordinate = solution[index + step_size]

    x_dist = future_coordinate[0] - coordinate[0]
    y_dist = coordinate[1] - future_coordinate[1]

    axis.quiver(coordinate[0], coordinate[1], x_dist, y_dist, color='r', scale=150, width=0.005)

    index+=step_size
  return

def gauss_elimination(a_matrix,b_matrix):
    #adding some contingencies to prevent future problems 
    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("ERROR: Squarematrix not given!")
        return
    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("ERROR: Constant vector incorrectly sized")
        return
     
    #initialization of nexessary variables
    n=len(b_matrix)
    m=n-1
    i=0
    j=i-1
    x=np.zeros(n)
    new_line="\n"
    
    #create our augmented matrix throug Numpys concatenate feature
    augmented_matrix = np.concatenate((a_matrix, b_matrix,), axis=1)
    print("the initial augmented matrix is: {x}{y}".format(x=new_line, y=augmented_matrix))
    print("solving for the upper-triangular matrix:")
    
    
    #applying gauss elimination:
    while i<n:
        
        # Partial Pivoting
        for p in range(i+1,n):
          if abs(augmented_matrix[i,i]) < abs(augmented_matrix[p][i]):
            augmented_matrix[[p,i]] = augmented_matrix[[i,p]]

        if augmented_matrix[i][i]==0.0: #fail-safe to eliminate divide by zero erroor!
            print("Divide by zero error")
            return
        for j in range(i+1,n):
            scaling_factor=augmented_matrix[j][i]/augmented_matrix[i][i]
            augmented_matrix[j]=augmented_matrix[j]-(scaling_factor * augmented_matrix[i])
            print(augmented_matrix) #not needed, but nice to visualize the process
            
        i=i+1
    
        #backwords substitution!
        x[m]=augmented_matrix[m][n]/augmented_matrix[m][m]
        for k in range(n-2,-1,-1):
            x[k]=augmented_matrix[k][n]
            for j in range(k+1,n):
                x[k]=x[k] - augmented_matrix[k][k]

            x[k] = x[k] / augmented_matrix[k][k] 
    
    #displaying solution 
    print("The following x vector matrix solves the above augmented matrix:")
    for answer in range(n):
        print("x{x} is {y}".format(x=answer,y=x[answer]))

    return n


def cubic_spline(axis, solution):
  # Cubic Spline Interpolation -> 3 degrees of freedom
  # Between each step size we can create a function of P1(x), P2(x), ... Pn(x)
  # n # of interpolating splines (always 1 less than the number of data points we're using)

  index = 0
  step_size = 20
  while index < 1:

    coordinate = solution[index]
    interior_coordinate = solution[index + step_size]
    future_coordinate = solution[index + 2 * step_size]

    cubic = np.zeros((8, 8))
    # p1(x) = ax^3 + bx^2 + cx + d
    x1, y1 = coordinate[0], coordinate[1]
    cubic[0] = [x1**3, x1**2, x1, 1, 0, 0, 0, 0]
    x2, y2 = interior_coordinate[0], interior_coordinate[1]
    cubic[1] = [x2**3, x2**2, x2, 1, 0, 0, 0, 0]
    x3, y3 = future_coordinate[0], future_coordinate[1]
    cubic[2] = [0, 0, 0, 0, x2**3, x2**2, x2, 1]
    cubic[3] = [0, 0, 0, 0, x3**3, x3**2, x3, 1]

    # ---- derivatives of smoothness ---- #
    # p1'(x) = 3ax^2 + 2bx + c
    # p1'(x) = p2'(x)
    cubic[4] = [-3 * (x2**2), -2 * (x2), -1, 0, 3 * (x2**2), 2 * (x2), 1, 0]
    # p1''(x) = 6ax + 2b
    # p1''(x) = p2''(x)
    cubic[5] = [-6 * x2, -2, 0, 0, 6 * x2, 2, 0, 0]

    # ---- normal interpolation ---- #
    cubic[6] = [6 * x1, 2, 0, 0, 0, 0 , 0, 0]
    cubic[7] = [0, 0, 0, 0, 6 * x3, 2 , 0, 0]

    cubic_1 = np.array([[x1**3, x1**2, x1, 1, 0, 0, 0, 0],
                        [x2**3, x2**2, x2, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, x2**3, x2**2, x2, 1],
                        [0, 0, 0, 0, x3**3, x3**2, x3, 1],
                        [-3 * (x2**2), -2 * (x2), -1, 0, 3 * (x2**2), 2 * (x2), 1, 0],
                        [-6 * x2, -2, 0, 0, 6 * x2, 2, 0, 0],
                        [6 * x1, 2, 0, 0, 0, 0 , 0, 0],
                        [0, 0, 0, 0, 6 * x3, 2 , 0, 0]])

    output = np.array([[y1],
                      [y2],
                      [y2],
                      [y3],
                      [0],
                      [0],
                      [0],
                      [0]])
    
    # print(cubic)
    # print(output)
    index+=1  


  x = list()
  y = list()
  t = list()

  x_ = list()
  y_ = list()

  i = 0  
  while i < len(solution) - step_size:
    coor = solution[i]
    # print(coor)
    x_temp = float(coor[0]) / 40
    y_temp = (800 - float(coor[1])) / 40
    t_temp = coor[2]
    # if x_temp not in x and y_temp not in y:

    x_.append(x_temp)
    y_.append(y_temp)

    x.append(coor[0])
    y.append(coor[1])
    t.append(coor[2])
    i+= step_size

  x_arr = np.array(x)
  y_arr = np.array(y)
  t_arr = np.array(t)

  x_t = np.array(x_)
  y_t = np.array(y_)

  x_cubic = CubicSpline(x=t_arr, y=x_arr, bc_type="natural")
  y_cubic = CubicSpline(x=t_arr, y=y_arr, bc_type="natural")
  t_interp = np.linspace(np.min(t_arr), np.max(t_arr), 50)

  x_map = CubicSpline(x=t_arr, y=x_t, bc_type="natural")
  y_map = CubicSpline(x=t_arr, y=y_t, bc_type="natural")

  axis.plot(x_cubic(t_interp), y_cubic(t_interp), "red")

  # y_cubic = CubicSpline(x=x_arr, y=y_arr)
  # x_interp = np.linspace(np.min(x_arr), np.max(x_arr), 50)

  # axis.plot(x_interp, y_cubic(x_interp), "red")

  return x_map, y_map, x_cubic, y_cubic, t_arr

def main():
  print("Running A* Simulation . . .")

  start = time.time()

  # image_path = cv2.imread("/home/bkhwaja/vscode/catkin_wos/src/mushr/mushr_base/mushr_base/mushr_base/maps/mit/short-course-33.png", 0)
  image_path = cv2.imread("/home/bkhwaja/vscode/catkin_wos/src/simulations/maps/plain.png", 0)
  height, width = image_path.shape
  final_image = image_path.copy()
  # print(image_path.shape)

  adj_list = convert_to_nodes(image_path, height, width)
  fig, ax = plt.subplots(1, 4)

  #inflating the edges to have room for error in collisions
  for gnode in adj_list:
    if gnode.active and gnode.edge:
      # print("here")
      temp_id = gnode.getId()
      adj_list[temp_id + 1].active = False
      adj_list[temp_id + 2].active = False
      adj_list[temp_id + 3].active = False
      adj_list[temp_id + 4].active = False
      adj_list[temp_id + 5].active = False

      adj_list[temp_id - 1].active = False
      adj_list[temp_id - 2].active = False
      adj_list[temp_id - 3].active = False
      adj_list[temp_id - 4].active = False
      adj_list[temp_id - 5].active = False

      adj_list[temp_id + 8100].active = False
      adj_list[temp_id + 800 * 2].active = False

      adj_list[temp_id - 800].active = False
      adj_list[temp_id - 800 * 2].active = False

      adj_list[temp_id - 800 - 1].active = False
      adj_list[temp_id - 800 + 1].active = False
      adj_list[temp_id + 800 - 1].active = False
      adj_list[temp_id + 800 + 1].active = False

  create_adjlist(nodes_arr=adj_list)

  draw_node(image_path, x=482, y=547) # Starting Position
  draw_node(image=image_path, x=309, y=233) # Ending Position

  fig.set_size_inches(20, 6.5)
  fig.suptitle("A* - Heuristic Graph Traversal", fontweight="bold", fontsize=20)
  fig.tight_layout()
  ax[0].imshow(image_path)
  ax[0].set_title("Start and End Node", fontweight="bold")
  # ax[0].set(xlim=[200, 600], ylim=[600, 200])
  ax[0].set(xlim=[0, 800], ylim=[800, 0])
  ax[0].set_axis_off()
  ax[0].annotate("Start", xy=(482, 547), xytext=(530,547), arrowprops={}, fontsize=15, fontweight="bold")
  ax[0].annotate("End", xy=(309, 233), xytext=(221,233), arrowprops={}, fontsize=15, fontweight="bold")

  astar(image_path, adj_list)

  end_id = 0
  for gnode in adj_list:
    x , y = gnode.getCoordinates()
    if x == 309 and y == 233:
      end_id = gnode.getId()
      break

  # print(end_id)

  print("Printing Solution  . . . (Linear Interpolation)")

  solution = print_path(image_path, adj_list, end_id)
  # print(solution)

  ax[1].imshow(image_path)
  ax[1].set_title("Exploring and Linear Interpolation", fontweight="bold")
  ax[1].set(xlim=[0, 800], ylim=[800, 0])
  ax[1].set_axis_off()

  print("Printing Vector Trajectory . . .")
  generate_vector_trajectory(ax[2], solution)

  ax[2].imshow(image_path)
  ax[2].set_title("Vector Generation", fontweight="bold")
  ax[2].set(xlim=[0, 800], ylim=[800, 0])
  ax[2].set_axis_off()

  ax[3].imshow(final_image)
  ax[3].set_title("Cubic Spline Generation", fontweight="bold")
  ax[3].set(xlim=[0, 800], ylim=[800, 0])
  ax[3].set_axis_off()

  spline_x, spline_y, x_cubic, y_cubic, t_arr = cubic_spline(ax[3], solution)

  print("--------------Finished Execution--------------")

  end = time.time()
  execution_time = end - start
  print("Executation time : " + str(execution_time))
  

  # plt.figure(figsize=(14,12))
  # Uncomment if you want to see the plotted images
  plt.show()

  return solution, spline_x, spline_y, x_cubic, y_cubic, t_arr

if __name__ == "__main__":
  main()
