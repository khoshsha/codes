from numba import jit
from numba import cuda
import numba
import numpy as np
import pandas as pd
import math
from timeit import default_timer as timer
from numba import guvectorize

start = timer()

fileaddress = "C:/Users/sksha/Desktop/dem.ascii"
# address = fileaddress.replace("\\", "/")
df = pd.read_csv(fileaddress.strip("\u202a"), delimiter=" ", skiprows=6, header=None)
df2 = pd.read_csv(fileaddress.strip("\u202a"), delimiter=" ", nrows=6, header=None)
ncols = int(df2[1][0])
nrows = int(df2[1][1])
cellsize = float(df2[1][4])
nodatavalue = float(df2[1][5])
print(ncols, "ncols")
print(nrows, "nrows")

totalt = 5*3600  # s
tstart = 0
cfl = 0.9
lw = cellsize
A = cellsize * cellsize
initdt = 0.001  # s
maxdt = 60
d = 1
n = 0.01
threshhold = 0.001
Rmass=20
raintime = [1, 2, 3, 4, 5]  # h
rains = [20, 20, 20, 20, 20]  # mm/h

topography = []
for i in range(nrows):
    topography.append([])

for i in range(ncols):
    for j in range(nrows):
        if not math.isnan(df[i][j]):
            topography[j].append(float(df[i][j]))
        else:
            topography[j].append(float(0))

#print(topography)


topography = np.array(topography, dtype=float)
gradientsv = np.zeros((nrows-1, ncols), dtype=float)
gradientsh = np.zeros((nrows, ncols-1), dtype=float)
waterdepths = np.zeros((nrows, ncols), dtype=float)
newwaterdepths=np.zeros((nrows, ncols), dtype=float)

cellswithvalues = 0
for rows in range(nrows):
    for cols in range(ncols):
        if topography[rows, cols] != nodatavalue:
            cellswithvalues = cellswithvalues + 1


raintime = np.array(raintime, dtype=float)
rains = np.array(rains, dtype=float)

for i in range(len(raintime)):
    raintime[i] = float(raintime[i]) * 3600
    rains[i] = float(rains[i]) / 3600000
print(rains, "rains")


@jit(nopython=True)
def solve(waterdepths, newwaterdepths, gradientsv, gradientsh):
    dt = initdt
    t = 0
    rainlimit = 1
    while t < totalt:
        # gradients
        for rows in range(nrows):
            for cols in range(ncols - 1):
                if topography[rows, cols] != nodatavalue and topography[rows, cols + 1] != nodatavalue:
                    gradientsh[rows, cols] = (topography[rows, cols + 1] + waterdepths[rows, cols + 1] - topography[
                        rows, cols] - waterdepths[rows, cols]) / d
                else:
                    gradientsh[rows, cols] = 0
        for rows in range(nrows-1):
            for cols in range(ncols):
                if topography[rows, cols] != nodatavalue and topography[rows + 1, cols] != nodatavalue:
                    gradientsv[rows, cols] = (topography[rows + 1, cols] + waterdepths[rows + 1, cols] - topography[
                        rows, cols] - waterdepths[rows, cols]) / d
                else:
                    gradientsv[rows, cols] = 0
        #print(gradientsv, "v")
        #print(gradientsh, "h")
        if t >= raintime[rainlimit] and rainlimit != (len(raintime) - 1):
            R = rains[rainlimit]
            rainlimit = rainlimit + 1
        else:
            R = rains[rainlimit - 1]


        mindt = 10000
        for rows in range(nrows):
            for cols in range(ncols):
                if rows > 1690:
                    R = 0
                fluxsum = 0
                if topography[rows, cols] != nodatavalue:
                    # right
                    if cols != ncols - 1:
                        if topography[rows, cols + 1] != nodatavalue:
                            gradient = float(-gradientsh[rows, cols])
                            if threshhold > gradient > (-threshhold):
                                flux = 0
                            else:
                                flux = (((waterdepths[rows, cols + 1]) ** (5 / 3)) * gradient * n * lw) / (
                                        n * math.sqrt(abs(gradient)))
                        else:
                            flux = 0
                    else:
                        flux = 0
                    if flux != 0:
                        dtcfl = abs(cfl * (lw / (flux / lw * 1)))  # 1 is distance between cell centers
                        if dtcfl < mindt:
                            mindt = dtcfl
                    fluxsum = fluxsum + flux
                    #print(fluxsum, "1")
                    # left
                    if cols != 0:
                        if topography[rows, cols - 1] != nodatavalue:
                            gradient = float(gradientsh[rows, cols - 1])
                            if threshhold > gradient > (-threshhold):
                                flux = 0
                            else:
                                #print("hello")
                                #print(waterdepths[rows, cols - 1],"waterdepth")
                                #print(gradient,"gradient")
                                #print(math.sqrt(abs(gradient)),"sqrt")
                                #print(waterdepths,"waterdepths2",t)
                                flux = (((waterdepths[rows, cols - 1]) ** (5 / 3)) * gradient * n * lw) / (
                                        n * math.sqrt(abs(gradient)))
                        else:
                            flux = 0
                    else:
                        flux = 0
                    if flux != 0:
                        dtcfl = abs(cfl * (lw / (flux / lw * 1)))  # 1 is distance between cell centers
                        if dtcfl < mindt:
                            mindt = dtcfl
                    fluxsum = fluxsum + flux
                    #print(fluxsum, "2")
                    # top
                    if rows != 0:
                        if topography[rows - 1, cols] != nodatavalue:
                            gradient = float(gradientsv[rows-1, cols])
                            if threshhold > gradient > (-threshhold):
                                flux = 0
                            else:
                                flux = (((waterdepths[rows - 1, cols]) ** (5 / 3)) * gradient * n * lw) / (
                                        n * math.sqrt(abs(gradient)))
                        else:
                            flux = 0
                    else:
                        flux = 0
                    if flux != 0:
                        dtcfl = abs(cfl * (lw / (flux / lw * 1)))  # 1 is distance between cell centers
                        if dtcfl < mindt:
                            mindt = dtcfl
                    fluxsum = fluxsum + flux
                    #print(fluxsum, "3")
                    # bottom
                    if rows != nrows - 1:
                        if topography[rows + 1, cols] != nodatavalue:
                            gradient = float(-gradientsv[rows, cols])
                            if threshhold > gradient > (-threshhold):
                                flux = 0
                            else:
                                flux = (((waterdepths[rows + 1, cols]) ** (5 / 3)) * gradient * n * lw) / (
                                        n * math.sqrt(abs(gradient)))
                        else:
                            flux = 0
                    else:
                        flux = 0
                    if flux != 0:
                        dtcfl = abs(cfl * (lw / (flux / lw * 1)))  # 1 is distance between cell centers
                        if dtcfl < mindt:
                            mindt = dtcfl
                    fluxsum = fluxsum + flux
                    #print(fluxsum, "4")
                    if t == 0:
                        dt = initdt
                    else:
                        if mindt != 10000:
                            if mindt>maxdt:
                                dt = mindt
                            else:
                                dt=maxdt
                        else:
                            dt = maxdt
                    if dt < 0:
                        dt = 0
                    newwaterdepths[rows, cols] = waterdepths[rows, cols] + (dt * R) - ((dt * fluxsum) / A)
                    if newwaterdepths[rows, cols]<0:
                        newwaterdepths[rows, cols]=0

                    #boundary condition

                    #if rows==nrows-1 and cols==1:
                    #    newwaterdepths[rows, cols]=newwaterdepths[rows, cols]-0.1
                    #    if newwaterdepths[rows, cols]<0:
                    #        newwaterdepths[rows, cols]=0

            # dt = 1
        for i in range(nrows):
            for j in range(ncols):
                waterdepths[i,j] = newwaterdepths[i,j]
        #print(waterdepths)
        t = t + dt
        print(dt, "dt", "  ", t, "t")
        print(((((Rmass/3600000) * cellswithvalues * t) - np.sum(waterdepths)) / (Rmass * cellswithvalues * t)) * 100, "mass error (%)")


solve(waterdepths, newwaterdepths, gradientsv, gradientsh)
print(waterdepths, "waterdepth")
filepath = "C:/Users/sksha/Desktop/output.txt"
try:
    file = open(filepath, 'w')
    file.close()
except:
    pass
with open(filepath, 'a') as output:
    output.write("NCOLS %s\n" % ncols)
    output.write("NROWS %s\n" % nrows)
    output.write("XLLCORNER 0\n")
    output.write("YLLCORNER 0\n")
    output.write("CELLSIZE 1\n")
    output.write("NODATA_VALUE %s\n" % nodatavalue)
    np.savetxt(output, waterdepths)
print(((((Rmass/3600000) * cellswithvalues * totalt) - np.sum(waterdepths)) / (Rmass * cellswithvalues * totalt)) * 100, "mass error (%)")
print("without GPU:", timer() - start)
