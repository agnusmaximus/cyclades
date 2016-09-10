# sample call: python plotting/plot_time_loss.py plotting/plot_time_loss.setting "Matrix Completion 8 threads - Movielens 1m"

from __future__ import print_function
import sys
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import subprocess
from matplotlib.ticker import FormatStrFormatter
import pylab
import commands
import numpy as np

# Plot customization
f, ax = plt.subplots()
ax.set_xlabel("Time/s", fontsize=26)
ttl = ax.title
ttl.set_position([.5, 1.05])
ax.set_ylabel("Objective Value", fontsize=26)
ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(axis='both', which='major', labelsize=26, pad=7)
ax.tick_params(axis='both', which='minor', labelsize=26, pad=7)

partition_time_matcher = re.compile("Partition Time\(s\): (.*)")

def RunCommand(command):
    # We expect the output of the command to be of form:
    # Line 1: Partition Time(s): ...
    # Lines 2+: Epoch: %d\tTime: %f\tLoss:%f
    print("Running command: %s" % command)
    out = commands.getstatusoutput(command)[1]
    lines = out.split("\n")
    partitioning_time = float(partition_time_matcher.search(lines[0]).group(1))
    times, losses = [], []
    for line in lines[1:]:
        epoch, time, loss = [x for x in line.split("\t") if x != ""]
        epoch = int(epoch.split(":")[-1].strip())
        time = float(time.split(":")[-1].strip())
        loss = float(loss.split(":")[-1].strip())
        times.append(time)
        losses.append(loss)
    times = [x + partitioning_time for x in times]
    return times, losses

def GetSettings(fname):
    # We expect the following format:
    # "command" curve_title curve_color
    f = open(fname)
    settings = []
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        setting = {"command" : "", "plot_title" : "", "plot_color" : ""}
        setting["command"] = elements[0]
        setting["plot_title"] = elements[1] # Curve title
        setting["plot_color"] = elements[2] # Curve color
        setting["line_style"] = elements[3]
        settings.append(setting)
    return settings

def Plot(name, settings):
    minx,maxx,miny,maxy = float("inf"),float("-inf"),float("inf"),float("-inf"),
    for setting in settings:
        # Run command and plot
        command = setting["command"]
        times, losses = RunCommand(command)
        ax.plot(times, losses, color=setting["plot_color"], linewidth=8, label=setting["plot_title"], linestyle=setting["line_style"])

        # Bounds tuning.
        minx=min(minx,min(times))
        maxx=max(maxx,max(times))
        miny=min(miny,min(losses), 100**int(np.log10(min(losses))))
        maxy=max(maxy,max(losses))

    # scale
    maxy = min(maxy, 10**20)
    ax.axis([minx*.95,maxx*1.05,miny*.95,maxy*1.05])

    # Do some stuff to make things nice looking.
    larger_power_10 = 10 ** int(np.log10(miny*.95))+1
    if larger_power_10 > maxy*1.05:
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

    basex = basey = 100
    ax.xaxis.set_major_locator(ticker.LogLocator(base = basex))
    ax.yaxis.set_major_locator(ticker.LogLocator(base = basey))
    ax.legend(loc="upper right")
    plt.savefig(name, bbox_inches='tight')

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: plot_time_loss.py settings_file output_name")
        sys.exit(0)
    ax.set_title(sys.argv[2], fontsize=26)
    settings = GetSettings(sys.argv[1])
    Plot(sys.argv[2], settings)
