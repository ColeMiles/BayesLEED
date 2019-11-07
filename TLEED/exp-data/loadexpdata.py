import matplotlib.pyplot as plt
import numpy as np

# Lol hardcoded

with open("Data.LaNiO3-10uc-1x1-7IntBeams-LT325eV", "r") as f:
    lines = f.readlines()
    del lines[0:5]
    line = lines[0]
    curves = []
    
    l = 0
    while l < len(lines):
        Is = []
        Vs = []
        while not line.startswith("("):
            I, V = map(float, line.split())
            Is.append(I)
            Vs.append(V)
            l += 1
            if l >= len(lines):
                break
            line = lines[l]
        l += 2
        if l >= len(lines):
            break
        line = lines[l]
        curves.append([Is, Vs])

for curve in curves:
    plt.plot(curve[0], curve[1])

plt.xlabel(r"$E$")
plt.ylabel(r"$I$")
plt.show()
