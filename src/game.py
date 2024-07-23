import pygame
import astar
import dijkstras
import rrt
import numpy as np

def draw_node(screen, pose):
   pygame.draw.rect(screen, "purple", [pose[0], pose[1], 5, 5])
   pygame.display.update()
   return

def main():
  pygame.init()

  screen = pygame.display.set_mode([800,800])

  # set the pygame window name
  pygame.display.set_caption('MIT Corridor Image')

  image = pygame.image.load("/home/bkhwaja/vscode/catkin_wos/src/mushr/mushr_base/mushr_base/mushr_base/maps/mit/short-course-33.png").convert()

  # cropped = pygame.Surface((600, 600))
  # cropped.blit(image, (200, 200), (200, 200, 400,400))

  screen.blit(image, (0,0))

  running = True
  path_run = True

  pygame.display.flip()

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

    draw_node(screen, solution[0])
    draw_node(screen, solution[len(solution) - 1])

    # while path_run :

    #   clock.tick(120)
      
    #   if value < len(solution):
    #     coor = solution[value]
    #     pygame.draw.rect(screen, "red", [coor[0], coor[1], 2, 2])
    #     pygame.display.update()
    #     value += 1
    #   else :
    #     path_run = False

    path_run = True
    value = 0
    while path_run :

      clock.tick(60)
      
      if value < len(t_interp):
        t = t_interp[value]
        # print(float(x_cubic(t)), float(y_cubic(t)))
        pygame.draw.rect(screen, "red", [float(x_cubic(t)), float(y_cubic(t)), 2, 2])
        pygame.display.update()
        value += 1
      else :
        path_run = False

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
         
         

    # Draw a solid blue circle in the center
    # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

    # Flip the display
    # pygame.display.flip()

  # Done! Time to quit.
  pygame.quit()
  return


if __name__ == "__main__":
  main()