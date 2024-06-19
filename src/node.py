class Node:

  def __init__(self, active, start, edge, x, y, id, distance, adjlist, pd, h):
    self.active = active
    self.edge = edge
    self.start = start
    self.x = x
    self.y = y
    self.id = id
    self.distance = distance
    self.adjlist = adjlist
    self.pd = pd
    self.h = h
  
  def __lt__(self, other):
    return self.distance < other.distance

  def isEdge(self):
    return self.edge
  
  def isStart(self):
    return self.start
  
  def getCoordinates(self):
    return self.x, self.y
  
  def getId(self):
    return self.id
  
  def getDistance(self):
    return self.distance
  
  def getAdjlist(self):
    return self.adjlist
  
  def setStart(self, start):
    self.start = start

  def setDistance(self, distance):
    self.distance = distance

  def setAdjlist(self, adjlist):
    self.adjlist = adjlist

  