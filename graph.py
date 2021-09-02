import matplotlib.pyplot as plt
from fmath import around

with open ("losses.csv") as file:
  data = file.read ()

losses = data.split (",") [:-1]

x = [i for i in range (len (losses))]
y = [float (c) for c in losses]

ymin = int (around (min (y), nearest = 10, direction = "down"))
ymax = int (around (max (y), nearest = 10, direction = "up"))

yticks = [i for i in range (ymin, ymax, 10)]

plt.figure ()
plt.scatter (x, y)
plt.yticks (yticks, [str (tick) for tick in yticks])
plt.show ()