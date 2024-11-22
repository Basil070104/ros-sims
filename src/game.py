import pygame
from pygame import mixer
import astar
import dijkstras
import rrt
import numpy as np

def draw_node(screen, pose, color):
   pygame.draw.rect(screen, color, [pose[0], pose[1], 10, 10])
   pygame.display.update()
   return

def display_text(screen, text, x, y, fontsize):
  # create a font object.
  # 1st parameter is the font file
  # which is present in pygame.
  # 2nd parameter is size of the font
  font = pygame.font.Font('freesansbold.ttf', fontsize)
  
  # create a text surface object,
  # on which text is drawn on it.
  text = font.render(text, True, (0, 0, 0), (255, 255, 255))
  
  # create a rectangular object for the
  # text surface object
  textRect = text.get_rect()
  
  # set the center of the rectangular object.
  textRect.center = (x, y)

  screen.blit(text, textRect)
  return

def main():
  print("---Welcome to Global Path Planning and Trajectory Mapping---")

  pygame.init()
  # mixer.init()

  mixer.music.load("../music/Ruben Had A Little Lamb.mp3")

  # Setting the volume 
  # mixer.music.set_volume(0.7) 
    
  # Start playing the song 
  # mixer.music.play() 

  screen = pygame.display.set_mode([800,800])

  image = pygame.image.load("/home/bkhwaja/vscode/catkin_wos/src/mushr/mushr_base/mushr_base/mushr_base/maps/mit/short-course-33.png").convert()

  # set the pygame window name
  pygame.display.set_caption('MIT Corridor Image')
  # cropped = pygame.Surface((600, 600))
  # cropped.blit(image, (200, 200), (200, 200, 400,400))

  screen.blit(image, (0,0))
  # Drawing Rectangle
  pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 650, 800, 150))
  display_text(screen, "Start", 100, 700, 15)
  draw_node(screen, (100, 720), "green")
  display_text(screen, "Destination", 100, 760, 15)
  draw_node(screen, (100, 780), "red")
  display_text(screen, "Dijkstra", 250, 700, 15)
  draw_node(screen, (250, 720), "blue")
  display_text(screen, "A*", 250, 760, 15)
  draw_node(screen, (250, 780), (112, 41, 99))

  pygame.display.flip()

  running = True
  path_run = True
  print_path = True

  solution, spline_x, spline_y, x_cubic, y_cubic, t_arr = astar.main()
  solution_d, spline = dijkstras.main()
  # solution, spline = rrt.main()

  t_interp = np.linspace(np.min(t_arr), np.max(t_arr), 250)

  while running :
    
    pose = None
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONUP:
           pose = pygame.mouse.get_pos()
          #  draw_node(screen, pose)
          #  print(pose)

    # Fill the background with white
    # screen.fill((255, 255, 255))

    ## display spline path for each algorithm
    clock = pygame.time.Clock()
    value = 0

    draw_node(screen, solution[0], "green")
    draw_node(screen, solution[len(solution) - 1], "red")

    # while path_run :

    #   clock.tick(120)
      
    #   if value < len(solution):
    #     coor = solution[value]
    #     pygame.draw.rect(screen, "red", [coor[0], coor[1], 2, 2])
    #     pygame.display.update()
    #     value += 1
    #   else :
    #     path_run = False

    if print_path :
      path_run = True
      value = 0
      print("Pygame A* Loading . . .")
      while path_run :

        clock.tick(60)
        
        if value < len(t_interp):
          t = t_interp[value]
          # print(float(x_cubic(t)), float(y_cubic(t)))
          pygame.draw.rect(screen, (112, 41, 99), [float(x_cubic(t)), float(y_cubic(t)), 2, 2])
          pygame.display.update()
          value += 1
        else :
          path_run = False


      print("Pygame Dijkstra Loading . . .")
      path_run = True
      value = 0
      while path_run :

        clock.tick(120)
        
        if value < len(solution_d):
          coor = solution_d[value]
          pygame.draw.rect(screen, "blue", [coor[0], coor[1], 2, 2])
          pygame.display.update()
          value += 1
        else :
          path_run = False

      print_path = False
         
         

    # Draw a solid blue circle in the center
    # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

    # Flip the display
    # pygame.display.flip()

  # Done! Time to quit.
  pygame.quit()
  return


if __name__ == "__main__":
  main()