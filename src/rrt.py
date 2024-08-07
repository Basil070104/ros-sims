import cv2
import node2
import math
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d
import numpy as np

# Samplin-Based Algorithm - reduces the number of nodes calculated
 
# 1. Sample a random position from the obstacle-free region of the map
# 2. Create a node that's associated with random position
# 3. Find a node already in the tree that's closest to the random position
# 4. Calculate a path from the random position to the position of the node
# 5. Continue to the next iteration if the path collides with something
# 6. Insert the node that's associated with the random position into the tree with the node (the node nearest to it) as its parent node.
# 7. Return the tree once the random position is within some distance of the goal position

# Setting the number of iterations to 1000

end_x = 309
end_y = 233

def draw_node(image, x, y):

  start_x = x - 2
  start_y = y - 2

  x = start_x
  y = start_y

  for i in range(0,5):
    for j in range(0,5):
      image[y][x] = 0
      x += 1
    x = start_x
    y += 1

def calculate_abs_distance(curr, other):
  x_dist = curr.x - other.x
  y_dist = curr.y - other.y

  dist = math.sqrt(x_dist**2 + y_dist**2)
  return dist

def calculate_abs_distance_end(curr):
  x_dist = curr.x - end_x
  y_dist = curr.y - end_y

  dist = math.sqrt(x_dist**2 + y_dist**2)
  return dist

def convert_to_nodes(image, height, width):

  nodes = list()
  end_x = 309
  end_y = 233

  id = 0
  for i in range(0, width):
    for j in range(0, height):
      if image[i][j] == 255:
        # h = calculate_heuristic(j , i, end_x, end_y)
        h = 0
        nodes.append(node2.Node2(True, False, False, j, i, id, float('inf'), [], -1, h, float('inf')))
        id += 1
      elif image[i][j] == 0:
        # h = calculate_heuristic(j , i, end_x, end_y)
        h = 0
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
        addEdge(adjlist, nodes_arr, temp_id - 800 - 1, math.sqrt(2))
        addEdge(adjlist, nodes_arr, temp_id - 800 + 1, math.sqrt(2))
        addEdge(adjlist, nodes_arr, temp_id + 800 - 1, math.sqrt(2))
        addEdge(adjlist, nodes_arr, temp_id + 800 + 1, math.sqrt(2))

        # addEdge(adjlist, nodes_arr, temp_id - 800 - 1, 2)
        # addEdge(adjlist, nodes_arr, temp_id - 800 + 1, 2)
        # addEdge(adjlist, nodes_arr, temp_id + 800 - 1, 2)
        # addEdge(adjlist, nodes_arr, temp_id + 800 + 1, 2)
      except:
        print(temp_id)
      
      gnodes.setAdjlist(adjlist)
      # if gnodes.id == 438082:
      #   print("id : {a} adjlist : {b}".format(a=gnodes.id, b=gnodes.adjlist))

def random_node(adjlist):
  valid = False
  rand = None

  while not valid :
   rand = random.choice(adjlist)
   if rand.active and not rand.edge and not rand.start:
     valid = True
  
  return rand

def nearest_node(new_node, valid_nodes):

  nearest_node = None
  distance = float('inf')

  for valid in valid_nodes:
    temp_dist = calculate_abs_distance(new_node, valid)
    if(temp_dist < distance):
      distance = temp_dist
      nearest_node = valid
  
  return nearest_node, distance

def generate_vector_trajectory(axis, solution):

  index = 0
  step_size = 1
  while index < len(solution) - step_size:
    # print("here")

    coordinate = solution[index]
    future_coordinate = solution[index + step_size]

    x_dist = future_coordinate[0] - coordinate[0]
    y_dist = coordinate[1] - future_coordinate[1]

    axis.quiver(coordinate[0], coordinate[1], x_dist, y_dist, color='r', scale=150, width=0.005)

    index+=step_size
  return

def cubic_spline(axis, solution):
  # Cubic Spline Interpolation -> 3 degrees of freedom
  # Between each step size we can create a function of P1(x), P2(x), ... Pn(x)
  # n # of interpolating splines (always 1 less than the number of data points we're using)

  index = 0
  step_size = 1
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
  i = 0  
  # solution.sort(axis=1)
  solution_arr = np.array(solution)
  solution_arr.sort(axis=0)
  print(solution_arr)
  x_arr_temp = solution_arr[:, 0]
  y_arr_temp = solution_arr[:, 1]
  while i < len(solution_arr) - step_size:
    # coor = solution_arr[i]
    # print(coor)
    x_temp = x_arr_temp[i]
    y_temp = y_arr_temp[i]
    if x_temp not in x and y_temp not in y:
      x.append(x_arr_temp[i])
      y.append(y_arr_temp[i])
    i+= step_size

  x_arr = np.array(x)
  y_arr = np.array(y)

  print(x_arr)
  print(np.any(x_arr[1:] <= x_arr[:-1]))
  print(y_arr)
  y_cubic = interp1d(x=x_arr, y=y_arr, kind="cubic", assume_sorted=False)
  x_interp = np.linspace(np.min(x_arr), np.max(x_arr), 50)

  axis.plot(x_interp, y_cubic(x_interp), "red")

  return y_cubic

def rrt(image, nodes_arr, numIterations, end_id, ax):

  max_distance = 75 # change number to optimize the algo better
  nodes = []

  start_id = 0
  for gnode in nodes_arr:
    x , y = gnode.getCoordinates()
    if x == 482 and y == 547:
      start_id = gnode.getId()
      break

  nodes.append(nodes_arr[start_id])
  j = 1

  for i in range(0, numIterations):
    in_dist = False
    while not in_dist:
      x_new = random_node(adjlist=nodes_arr)

      # x_dist = float('inf')
      # near_node = None
      # for node in nodes:
      #   temp = calculate_abs_distance(node, x_new)
      #   if temp < x_dist:
      #     x_dist = temp
      #     near_node = node
      near_node , x_dist = nearest_node(x_new, nodes)
      # x_dist = calculate_abs_distance(nodes_arr[start_id], x_new)
      if x_dist <= max_distance:
        in_dist = True
        draw_node(image, x_new.x, x_new.y)

        # near_node = nearest_node(x_new, nodes)
        nodes.append(x_new)
        x_new.pd = near_node.id

        print("Added Node : {x}, {y}".format(x=x_new.x, y=x_new.y))
        ax.plot([near_node.x, x_new.x], [near_node.y, x_new.y], 'g-')

        # plt.imshow(image)
        # plt.pause(0.0001)

        # check if we should connect it to the last node
        end_dist = calculate_abs_distance_end(x_new)
        if end_dist <= max_distance:
          ax.plot([near_node.x, end_x], [near_node.y, end_y], 'g-')
          nodes_arr[end_id].pd = x_new.id
          return
    
    # if j > 5: 
    #   plt.imshow(image)
    #   plt.pause(0.0001)
    #   j = 0
    # j+=1

def label_path(image, nodes_arr, end_id, ax):

  solution = list()
  curr_id = end_id
  i = 0
  while curr_id != -1:
    x, y = nodes_arr[curr_id].getCoordinates()
    image[y][x] = 50

    temp = nodes_arr[curr_id].pd 
    # print(temp)
    solution.insert(0, [x, y, 0])

    if temp != -1:
      ax.plot([nodes_arr[curr_id].x, nodes_arr[temp].x], [nodes_arr[curr_id].y, nodes_arr[temp].y], 'r-')
    curr_id = temp

    # if i > 50:
    #   plt.imshow(image)
    #   plt.pause(0.0001)
    #   i = 0
    # i+=1

  return solution

def main():
  print("Running RRT Simulation . . .")

  image_path = cv2.imread("/home/bkhwaja/vscode/catkin_wos/src/mushr/mushr_base/mushr_base/mushr_base/maps/mit/short-course-33.png", 0)
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

      adj_list[temp_id + 800].active = False
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
  fig.suptitle("RRT - Rapidly Exploring Random Tree", fontweight="bold", fontsize=20)
  fig.tight_layout()
  ax[0].imshow(image_path)
  ax[0].set_title("Start and End Node", fontweight="bold")
  ax[0].set(xlim=[200, 600], ylim=[600, 200])
  ax[0].set_axis_off()
  ax[0].annotate("Start", xy=(482, 547), xytext=(530,547), arrowprops={}, fontsize=15, fontweight="bold")
  ax[0].annotate("End", xy=(309, 233), xytext=(221,233), arrowprops={}, fontsize=15, fontweight="bold")

  end_id = 0
  for gnode in adj_list:
    x , y = gnode.getCoordinates()
    if x == 309 and y == 233:
      end_id = gnode.getId()
      break

  rrt(image_path, adj_list, 1000, end_id, ax[1])

  print("Printing Solution . . .")
  solution = label_path(image_path, adj_list, end_id, ax[1])

  ax[1].imshow(image_path)
  ax[1].set_title("Exploring and Linear Interpolation", fontweight="bold")
  ax[1].set(xlim=[200, 600], ylim=[600, 200])
  ax[1].set_axis_off()

  print("Printing Vector Trajectory . . .")
  # print(solution)
  generate_vector_trajectory(ax[2], solution)

  ax[2].imshow(image_path)
  ax[2].set_title("Vector Generation", fontweight="bold")
  ax[2].set(xlim=[200, 600], ylim=[600, 200])
  ax[2].set_axis_off()

  ax[3].imshow(final_image)
  ax[3].set_title("Cubic Spline Generation", fontweight="bold")
  ax[3].set(xlim=[200, 600], ylim=[600, 200])
  ax[3].set_axis_off()

  spline = cubic_spline(ax[3], solution)

  plt.show()

  return solution, spline

if __name__ == "__main__":
  main()