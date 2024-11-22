import numpy as np
import cv2
import matplotlib.pyplot as plt
import node
import heapq
import math
from scipy.interpolate import interp1d

# Start Postion : X = 482, Y = 547
# End Position : X = 309, Y = 233
# Adjacency list with (id, cost)
# Starting Id : 438082


# Start and End Nodes have a dimension of 5 x 5
# x and y are the center of the nodes
def draw_node(image, x, y):

  start_x = x - 2
  start_y = y - 2

  # image[y][x] = 100

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

  id = 0
  for i in range(0, width):
    for j in range(0, height):
      if image[i][j] == 255:
        nodes.append(node.Node(True, False, False, j, i, id, float('inf'), [], -1, 0))
        id += 1
      elif image[i][j] == 0:
        nodes.append(node.Node(True, False, True, j , i, id, float('inf'), [], -1, 0))
        id += 1
      else:
        nodes.append(node.Node(False, False, False, j , i, id, float('inf'), [], -1, 0))
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
      except:
        print(temp_id)
      
      gnodes.setAdjlist(adjlist)
      # if gnodes.id == 438082:
      #   print("id : {a} adjlist : {b}".format(a=gnodes.id, b=gnodes.adjlist))
      


def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at (X={a}, Y={b})".format(a=x, b=y))
      

def dijkstras(image, nodes_arr):
  pq = []
  arr_len = len(nodes_arr)
  visited = [False] *  arr_len
  
  # for i in range(0, arr_len):
  #   heapq.heappush(pq, nodes_arr[i])

  # inflate the edges to avoid collisions
  # 3 nodes left and right

  start_id = 0
  for gnode in nodes_arr:
    x , y = gnode.getCoordinates()
    if x == 482 and y == 547:
      start_id = gnode.getId()
      break

  
  # print(start_id)
  # print(arr_len)
  

  # set the starting point in the pq
  nodes_arr[start_id].setDistance(0)
  heapq.heappush(pq, nodes_arr[start_id])

  # print(pq[0].getDistance())
  i = 0

  while len(pq) != 0:
  # for i in range(0,100):
    gnode = heapq.heappop(pq)
    id = gnode.id
    distance = gnode.distance
    visited[id] = True
    # print("id popped : {id} and distance : {distance}".format(id=id, distance=distance))

    adj_list = nodes_arr[id].getAdjlist()
    # print(adj_list)

    for adjlink in adj_list:
      if not visited[adjlink[0]]:
        curr_dis = nodes_arr[adjlink[0]].distance
        if distance + adjlink[1] < curr_dis:
          nodes_arr[adjlink[0]].distance = distance + adjlink[1]
          nodes_arr[adjlink[0]].pd = id

          x, y = nodes_arr[adjlink[0]].getCoordinates()
          image[y][x] = 0
          heapq.heappush(pq, nodes_arr[adjlink[0]])

          if x == 309 and y == 233:
            return
    

    # if i > 2000: 
    #   plt.imshow(image)
    #   plt.pause(0.0001)
    #   i = 0

    # i+=1
        # if not any(item.id == for item in pq)

    # for i in pq:
    #   print(i.id, i.distance)

  # print(pq[0].id, pq[1].id, pq[2].id, pq[3].id, pq[4].id, pq[5].id)

def print_path(image, node_arr, end_id):

  solution = list()
  curr_id = end_id
  i = 0
  while curr_id != -1:
    x, y = node_arr[curr_id].getCoordinates()
    image[y][x] = 200

    curr_id = node_arr[curr_id].pd 
    solution.insert(0, [x, y, 0])

    # if i > 50:
    #   plt.imshow(image)
    #   plt.pause(0.0001)
    #   i = 0
    # i+=1
  
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

def onclick(event):
    print('Clicked at x = {x}, y = {y}'.format(x=event.xdata, y=event.ydata))
    return event.xdata, event.ydata

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
  while i < len(solution) - step_size:
    coor = solution[i]
    # print(coor)
    x_temp = coor[0]
    y_temp = coor[1]
    if x_temp not in x and y_temp not in y:
      x.insert(0, coor[0])
      y.insert(0, coor[1])
    i+= step_size

  x_arr = np.array(x)
  y_arr = np.array(y)

  # print(x_arr)
  # print(np.any(x_arr[1:] <= x_arr[:-1]))
  # print(y_arr)
  y_cubic = interp1d(x=x_arr, y=y_arr, kind="cubic", assume_sorted=False)
  x_interp = np.linspace(np.min(x_arr), np.max(x_arr), 50)

  axis.plot(x_interp, y_cubic(x_interp), "red")

  return y_cubic

def main():
    image_path = cv2.imread("/home/bkhwaja/vscode/catkin_wos/src/simulations/maps/short-course-33.png", 0)
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

    # for row in adj_list:
    #   for nodes in row:
    #     print("{coor} and id : {id}".format(coor=nodes.getCoordinates(), id=nodes.getId()))

    draw_node(image_path, x=482, y=547) # Starting Position
    draw_node(image=image_path, x=309, y=233) # Ending Position
    # fig, ax = plt.subplots()
    # fig.canvas.mpl_connect("mouse event clicked", onclick)

    print("Running Dijkstra's Simulation . . .")

    fig.set_size_inches(20, 6.5)
    fig.suptitle("Dijkstra - Base Generation", fontweight="bold", fontsize=20)
    fig.tight_layout()
    ax[0].imshow(image_path)
    ax[0].set_title("Start and End Node", fontweight="bold")
    ax[0].set(xlim=[200, 600], ylim=[600, 200])
    ax[0].set_axis_off()
    ax[0].annotate("Start", xy=(482, 547), xytext=(530,547), arrowprops={}, fontsize=15, fontweight="bold")
    ax[0].annotate("End", xy=(309, 233), xytext=(221,233), arrowprops={}, fontsize=15, fontweight="bold")

    dijkstras(image=image_path, nodes_arr=adj_list)

    end_id = 0
    for gnode in adj_list:
      x , y = gnode.getCoordinates()
      if x == 309 and y == 233:
        end_id = gnode.getId()
        break

    # print(end_id)

    print("Printing Solution  . . .")

    solution = print_path(image_path, adj_list, end_id)

    # print(solution)

    ax[1].imshow(image_path)
    ax[1].set_title("Exploring and Linear Interpolation", fontweight="bold")
    ax[1].set(xlim=[200, 600], ylim=[600, 200])
    ax[1].set_axis_off()

    print("Printing Vector Trajectory . . .")
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

    print("--------------Finished Execution--------------")

    # plt.figure(figsize=(14,12))

    # Uncomment if you want to see the plot
    # plt.show(block=False)

    # cv2.imshow('Path Simple', image_path)

    # cv2.setMouseCallback('Path Simple', get_mouse_coordinates)

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    return solution, spline

if __name__ == "__main__":
  main()


  