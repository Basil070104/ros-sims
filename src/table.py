import pandas as pd

def main():

  datat = {
    "Dijkstra" : [True, False, True, False],
    "A*" : [True, False, True, False],
    "RRT" : [True, False, True, False]
  }

  df = pd.DataFrame(data=datat, index=["1", "2", "3", "4"])

  print(df)
  return

if __name__ == "__main__":
  main()