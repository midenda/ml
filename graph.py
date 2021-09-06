import matplotlib.pyplot as plt
from fmath import around

with open ("losses.csv") as file:
  data = file.read ()

losses = data.split (",") [:-1]

x = [i for i in range (len (losses))]
y = [float (c) for c in losses]

# ymin = int (around (min (y), nearest = 10, direction = "down"))
# ymax = int (around (max (y), nearest = 10, direction = "up"))

# spacing = int (around (abs (ymax - ymin) / 10, nearest = 10))

# yticks = [i for i in range (ymin, ymax, spacing)]

plot_ratio = 0.78
plot_size = 9

figure, axis = plt.subplots (1, 1, gridspec_kw = {
                    'top': 0.95,
                    'bottom': 0.1,
                    'left': 0.1,
                    'right': 0.95,
                    'hspace': 0,
                    'wspace': 0.05
                }, figsize = (plot_size, plot_size * plot_ratio), num = 1)

plt.scatter (x, y, s = 50, label = "Loss", marker = ".", color = (0, 0, 0), linewidth = 1)
# plt.yticks (yticks, [str (tick) for tick in yticks])
plt.yscale ("log")

axis.set_ylabel ("$ Loss $", fontsize = 11)
axis.set_xlabel ("$ Iterations $", fontsize = 11)
axis.tick_params ("y", labelleft = True, labelright = False, labelsize = 10)

plt.show ()