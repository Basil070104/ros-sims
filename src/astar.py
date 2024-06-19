import node2
import cv2
import math
import matplotlib.pyplot as plt
import heapq

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

  # h = D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
  # where D is length of each node(usually = 1) and 
  # D2 is diagonal distance between each node (usually = sqrt(2) ).

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

def astar(image, nodes_arr):
  pq = []
  arr_len = len(nodes_arr)
  visited = [False] *  arr_len
  
  # for i in range(0, arr_len):
  #   heapq.heappush(pq, nodes_arr[i])

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
  nodes_arr[start_id].f = nodes_arr[start_id].distance + nodes_arr[start_id].h
  heapq.heappush(pq, nodes_arr[start_id])

  # print(pq[0].getDistance())
  i = 0

  while len(pq) != 0:
  # for i in range(0,100):
    gnode = heapq.heappop(pq)
    id = gnode.id
    distance = gnode.distance
    f = gnode.f
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
          nodes_arr[adjlink[0]].f = nodes_arr[adjlink[0]].distance + nodes_arr[adjlink[0]].h

          x, y = nodes_arr[adjlink[0]].getCoordinates()
          image[y][x] = 0
          heapq.heappush(pq, nodes_arr[adjlink[0]])

          if x == 309 and y == 233:
            return
    

    if i > 2000: 
      plt.imshow(image)
      plt.pause(0.0001)
      i = 0

    i+=1
        # if not any(item.id == for item in pq)

    # for i in pq:
    #   print(i.id, i.distance)

  # print(pq[0].id, pq[1].id, pq[2].id, pq[3].id, pq[4].id, pq[5].id)

def print_path(image, node_arr, end_id):

  solution = list()
  curr_id = end_id
  i = 0
  while curr_id != 438082:
    x, y = node_arr[curr_id].getCoordinates()
    image[y][x] = 200

    solution.insert(0, [x, y, 0])
    curr_id = node_arr[curr_id].pd 

    if i > 50:
      plt.imshow(image)
      plt.pause(0.0001)
      i = 0
    i+=1

  return solution

def main():
  print("Running A* Simulation . . .")

  image_path = cv2.imread("/home/bkhwaja/vscode/catkin_wos/src/mushr/mushr_base/mushr_base/mushr_base/maps/mit/short-course-33.png", 0)
  height, width = image_path.shape
  # print(image_path.shape)

  adj_list = convert_to_nodes(image_path, height, width)


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

  plt.figure(2, figsize=(14,12))
  plt.imshow(image_path)

  astar(image_path, adj_list)

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

  # plt.figure(figsize=(14,12))
  plt.imshow(image_path)
  plt.show()

  return solution

if __name__ == "__main__":
  main()
