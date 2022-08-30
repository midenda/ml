import matplotlib.pyplot as plt
from argparse import ArgumentParser
from fmath import around


colours = [(0.1, 0.1, 0.1), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]

# Parse command line arguments
parser = ArgumentParser ()
parser.add_argument (type = str, dest = "filename")
parser.add_argument ("--fit", action = "store_true", default = False, dest = "fit")

args = parser.parse_args ()

# Stream in data to plot from file
with open (args.filename) as file:
    data = file.read ()

filetype = args.filename.split (".")[1]

if filetype == "csv":
    series = []
    rows = data.split ("\n")

    if len (rows) > 1:
        for i in range (len (rows)):
            series.append ([float (c) for c in rows [i].split (",") [:-1]])
    else:
        values = data.split (",")
        series.append ([i for i in range (len (values))] [:-1]) # x values
        series.append ([float (c) for c in values] [:-1]) # y values

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

for i in range (1, len (series) - 1):
    plt.scatter (series [0], series [i], s = 10, label = "Loss", marker = "+", color = colours [i - 1], linewidth = 0.5)
# plt.yticks (yticks, [str (tick) for tick in yticks])

if args.fit:
    plt.plot (series [0], series [-1], linestyle = "-", color = (1, 0, 0))

# if (max (y) / min (y)) > 500:
# plt.yscale ("log")

axis.set_ylabel ("$ Loss $", fontsize = 11)
axis.set_xlabel ("$ Iterations $", fontsize = 11)
axis.tick_params ("y", labelleft = True, labelright = False, labelsize = 10)

plt.show ()