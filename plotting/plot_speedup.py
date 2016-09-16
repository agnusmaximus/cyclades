# sample call: python plotting/plot_speedup.py plotting/quick_nh2010_speedup.settings "Test Title" 0

from __future__ import print_function
import sys

import matplotlib
matplotlib.use('Agg')

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
import os

# Plot customization
f, ax = plt.subplots(figsize=(15,15))
ax.set_xlabel("n_threads", fontsize=26)
ttl = ax.title
ttl.set_position([.5, 1.05])
ax.set_ylabel("Speedup Over Hogwild Serial", fontsize=26)
# ax.set_xscale('log') # Make cyclades cache behavior look impressive.
ax.tick_params(axis='both', which='major', labelsize=26, pad=7)
ax.tick_params(axis='both', which='minor', labelsize=26, pad=7)

partition_time_matcher = re.compile("Partition Time\(s\): (.*)")
plot_cache_directory = "./plotting_cache/"


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
        setting["threads"] = [int(x) for x in elements[4].split("|")]
        setting["learning_rates"] = [float(x) for x in elements[5].split("|")]
        settings.append(setting)

        if "n_threads" in setting["command"]:
            print("Error: command must not have n_threads set")
            sys.exit(0)
        if "learning_rate" in setting["command"]:
            print("Error: command must not have learning rate set")
            sys.exit(0)
    return settings

def RunCommand(command, use_cached_output, reps=3):
    out = ""
    command_file = plot_cache_directory + "".join(x for x in command if x.isalnum())
    if not use_cached_output:
        # We expect the output of the command to be of form:
        # Line 1: Partition Time(s): ...
        # Lines 2+: Epoch: %d\tTime: %f\tLoss:%f
        print("Running command: %s" % command)
        for rep in range(reps):
            out += commands.getstatusoutput(command)[1]
        f_cache_out = open(command_file, "w+")
        print(out, file=f_cache_out);
        f_cache_out.close()
    else:
        print("Loading command file: %s" % command_file)
        f_cache_in = open(command_file, "r")
        out = f_cache_in.read().strip()
        f_cache_in.close()

    lines = out.split("\n")
    partition_time = 0
    epoch_time_data, epoch_loss_data = {}, {}
    n_epochs = 0
    for line in lines:
        partitioning_match = partition_time_matcher.search(line)
        if partitioning_match:
            partition_time = float(partitioning_match.group(1))
        else:
            epoch, time, loss = [x for x in line.split("\t") if x != ""]
            epoch = int(epoch.split(":")[-1].strip())
            time = float(time.split(":")[-1].strip())
            loss = float(loss.split(":")[-1].strip())
            if epoch not in epoch_time_data:
                epoch_time_data[epoch] = 0
            if epoch not in epoch_loss_data:
                epoch_loss_data[epoch] = 0
            epoch_time_data[epoch] += time + partition_time
            epoch_loss_data[epoch] += loss
            n_epochs = max(n_epochs, epoch)
    for epoch, value in epoch_time_data.items():
        epoch_time_data[epoch] = value / reps
    for epoch, value in epoch_loss_data.items():
        epoch_loss_data[epoch] = value / reps
    times = [epoch_time_data[x] for x in range(n_epochs)]
    losses = [epoch_loss_data[x] for x in range(n_epochs)]
    return times, losses

def Plot(name, settings, threads, use_cached_output):
    # [setting_index][num_thread][0] - times
    # [setting_index][num_thread][1] - losses
    individual_times_losses = {}
    for setting_index, setting in enumerate(settings):
        if setting_index not in individual_times_losses:
            individual_times_losses[setting_index] = {}
        for t_index, thread in enumerate(threads):
            command = setting["command"]
            command += " --n_threads=%d" % thread
            command += " --learning_rate=%s" % str(setting["learning_rates"][t_index])
            times, losses = RunCommand(command, use_cached_output)
            individual_times_losses[setting_index][thread] = [times, losses]

    # Get the loss that all threads of all settings achieve.
    global_loss = -1000000000
    for setting_index, setting in enumerate(settings):
        for thread in threads:
            global_loss = max(global_loss, min(individual_times_losses[setting_index][thread][1]))

    def time_to_target_loss(settings):
        for index, loss in enumerate(settings[1]):
            if loss <= global_loss:
                return settings[0][index]
        assert(False)
        return float("inf")

    # Get the baseline 1 threaded times.
    individual_times_to_target_losses = {}
    for setting_index, setting in enumerate(settings):
        time_to_target_losses = [time_to_target_loss(individual_times_losses[setting_index][t]) for t in threads]
        individual_times_to_target_losses[setting_index] = time_to_target_losses

    # Get speedups. Use setting_index=0 with thread=0 as baseline.
    individual_speedups = {}
    miny, maxy = float('inf'), float('-inf')
    for setting_index, setting, in enumerate(settings):
        speedup = [float(individual_times_to_target_losses[0][0]) / float(individual_times_to_target_losses[setting_index][i]) for i,t in enumerate(threads)]
        individual_speedups[setting_index] = speedup
        miny = min(speedup+[miny])
        maxy = max(speedup+[maxy])
    minx,maxx = min(threads), max(threads)

    for setting_index, setting in enumerate(settings):
        speedup = individual_speedups[setting_index]
        ax.plot(threads, speedup, color=setting["plot_color"], linewidth=8, label=setting["plot_title"], linestyle=setting["line_style"])

    # Styling
    ax.axis([minx*.95,maxx*1.05,miny*.95,maxy*1.05])
    inc = 1 if max(threads) <= 2 else 2
    xticks = list(range(1, max(threads)+1, inc))
    ax.set_xticks(xticks)
    ax.legend(loc="upper left", fontsize=22)

    plt.savefig(name, bbox_inches='tight')

if __name__=="__main__":
    if len(sys.argv) != 4:
        print("Usage: plot_speedup.py settings_file output_name use_cached_output")
        sys.exit(0)

    if not os.path.exists(plot_cache_directory):
        os.makedirs(plot_cache_directory)
    ax.set_title(sys.argv[2], fontsize=26)
    settings = GetSettings(sys.argv[1])
    threads = settings[0]["threads"]
    if 1 not in threads:
        threads = [1] + threads
    threads.sort()
    Plot(sys.argv[2], settings, threads, int(sys.argv[3]))
