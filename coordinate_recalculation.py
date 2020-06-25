#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:22:40 2019

This python script takes a pdb as input file.
It read all atom coordinates.
If set to look for solvent molecules it find solvent molecules that for a hydrogen bond with mol
    Count how many hydrogen bonds are there per frame and divide by molecule count
    Show in diagram how hydrogen bond count changes with time
Other thing it does is:
    Repostion main molecule center at coordinates x=0, y=0, z=0
    So that the mol belongs to the XY plane and carboxil grom is towards the postive X direction
    Then all other molecules whose center is within a certain distance coordinates ar recalculated
    to keep realtion of the new coordinates of the main melecule.
@author: auzins
"""

import glob, os#, shutil
from multiprocessing import Pool#, cpu_count
#import psutil
import json
import pickle
import traceback
from bisect import bisect_left, bisect_right
from datetime import datetime
import math
import scipy.stats as ss
#from scipy import stats
from matplotlib import cm
import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import matplotlib.ticker as tkr
from statsmodels.nonparametric.bandwidths import bw_scott#, bw_silverman, select_bandwidth
import scipy
import gc
#from sklearn.neighbors import KernelDensity

def to_energy(current, base):
    return np.multiply(-np.log(np.divide(current, base)), 8.31446261815324 * 300 / 1000)

def round_up_to_even(f):
    val = int(round(f / 2.) * 2)
    if val < 40:
        val = 40
    return val


def vonmises_pdf(x, mu, kappa):
    return np.exp(kappa * np.cos(x - mu)) / (2*np.pi * scipy.special.i0(kappa))

def vonmises_fft_sphere_kde(x, x_kappa, x_n_bins, x_min, x_max, y, y_kappa, y_n_bins, y_min, y_max):
    x_bins = np.linspace(x_min, x_max, x_n_bins + 1, endpoint=True)
    y_bins = np.linspace(y_min, y_max, y_n_bins + 1, endpoint=True)
    hist_n, x_bin_edges, y_bin_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    x_bin_centers = np.mean([x_bin_edges[1:], x_bin_edges[:-1]], axis=0)
    y_bin_centers = np.mean([y_bin_edges[1:], y_bin_edges[:-1]], axis=0)
    x_kernel = vonmises_pdf(
        x=x_bin_centers,
        mu=(x_min+x_max)/2,
        kappa=x_kappa
    )
    y_kernel = vonmises_pdf(
        x=y_bin_centers,
        mu=(y_min+y_max)/2,
        kappa=y_kappa
    )
    kernel = np.outer(x_kernel, y_kernel)
    kde = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(hist_n)))
    kde /= np.trapz(np.trapz(kde, y_bin_centers, axis=1), x_bin_centers, axis=0)
    return x_bin_centers, y_bin_centers, kde

def vonmises_fft_kde(data, kappa, n_bins, data_min, data_max):
    bins = np.linspace(data_min, data_max, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(data, bins=bins)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kernel = vonmises_pdf(
        x=bin_centers,
        mu=(data_min+data_max)/2,
        kappa=kappa
    )
    kde = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(hist_n)))
    kde /= np.trapz(kde, x=bin_centers)
    return bin_centers, kde

def scipy_kde(data, n_bins, data_min, data_max):
    bins = np.linspace(data_min, data_max, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(data, bins=bins)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kde = ss.gaussian_kde(data, bw_method='scott')
    return bin_centers, kde.evaluate(bin_centers)

def Spherical_KDE(name, title, xname, yname, x, xmin, xmax, y, ymin, ymax, cmapName, proportion):
    xbinWidth = bw_scott(x)*proportion
    ybinWidth = bw_scott(y)*proportion

    xbin = round_up_to_even((xmax - xmin)/xbinWidth)
    ybin = round_up_to_even((ymax - ymin)/ybinWidth)

    # Peform the kernel density estimate
    #xx, yy = np.mgrid[xmin:xmax:xbin*1j, ymin:ymax:ybin*1j]
    #positions = np.vstack([xx.ravel(), yy.ravel()])
    #values = np.vstack([x, y])

    xbin_centers, ybin_centers, kde_2d = vonmises_fft_sphere_kde(
        x, 10, xbin, xmin, xmax,
        y, 10, ybin, ymin, ymax)

    #kernel = ss.gaussian_kde(values)
    #f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(7, 6))
    grid = plt.GridSpec(4, 5, hspace=0.5, wspace=0.5)
    main_ax = fig.add_subplot(grid[1:, :-2])
    y_plot = fig.add_subplot(grid[1:, -2], xticklabels=[], sharey=main_ax)
    x_plot = fig.add_subplot(grid[0, :-2], yticklabels=[], sharex=main_ax)
    cbaxes = fig.add_subplot(grid[1:, -1])

    main_ax.set_xlim(xmin, xmax)
    main_ax.set_ylim(ymin, ymax)
    # Contourf plot

	# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    ti = 0
    tj = 0
    max_val = max(map(max, kde_2d))
    for col in kde_2d:
        for val in col:
            val = to_energy(val, max_val)
            kde_2d[ti][tj] = val
            tj += 1
        ti += 1
        tj = 0
    
    CS = main_ax.contourf(xbin_centers, ybin_centers, kde_2d.T, 50, cmap=cmapName)
    cb = fig.colorbar(CS, cax=cbaxes)
    cb.set_label("Density")
    # histogram on the attached axes
    xbin_centers, xkde = vonmises_fft_kde(x, 10, xbin, xmin, xmax)
    x_plot.plot(xbin_centers, xkde)

    ybin_centers, ykde = vonmises_fft_kde(y, 10, ybin, ymin, ymax)
    y_plot.plot(ykde, ybin_centers)
    main_ax.set_xlabel(xname)
    main_ax.set_ylabel(yname)
    fig.suptitle(title, fontsize=16)
    fig.savefig(name, format='pdf')
    fig.show()

#    #Set colours and render
#    fig = plt.figure(figsize=(10, 8))
#    ax = fig.add_subplot(111)
#    ax.scatter(x, y, edgecolors='none', s=10)
#    #ax.set_aspect("equal")
#    #ax.view_init(180, 30)
#    #ax.plot_wireframe(x, y, z, color="k") #not needed?!
#    plt.savefig(title+"_scatter.pdf", format='pdf')
#    plt.show() 

    n_theta = len(ybin_centers) # number of values for theta
    n_phi = len(xbin_centers)  # number of values for phi
    r = 2        #radius of sphere

    theta, phi = np.mgrid[0:np.pi:n_theta*1j, 0:2*np.pi:n_phi*1j]

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    # mimic the input array
    # array columns phi, theta, value
    # first n_theta entries: phi=0, second n_theta entries: phi=0.0315..

    inp = []
    for j in phi[0, :]:
        for i in theta[:, 0]:
            x_index = np.argmin(np.abs((xbin_centers-xbin_centers.min())*
                                       2*np.pi/(xbin_centers.max()-xbin_centers.min())-j))
            y_index = np.argmin(np.abs((ybin_centers-ybin_centers.min())*
                                       np.pi/(ybin_centers.max()-ybin_centers.min())-i))
            val = kde_2d.item(x_index, y_index).real
            inp.append(val)
    inp = np.array(inp)

    #reshape the input array to the shape of the x,y,z arrays.
    c = inp.reshape((n_phi, n_theta)).T
    c = (c-c.min())/(c.max()-c.min())

    #Set colours and render
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, facecolors=cm.viridis(c), alpha=0.5, linewidth=1)
    ax.set_xlim([-1.1*r, 1.1*r])
    ax.set_ylim([-1.1*r, 1.1*r])
    ax.set_zlim([-1.1*r, 1.1*r])
    #ax.set_aspect("equal")
    #ax.view_init(180, 30)
    #ax.plot_wireframe(x, y, z, color="k") #not needed?!
    plt.savefig(title+"_sphere.pdf", format='pdf')
    plt.show()
    
    
    

def Cylindrical_KDE(name, title, xname, yname, x, xmin, xmax, y, ymin, ymax, cmapName, proportion):
    xbinWidth = bw_scott(x)*proportion
    ybinWidth = bw_scott(y)*proportion

    xbin = round_up_to_even((xmax - xmin)/xbinWidth)
    ybin = round_up_to_even((ymax - ymin)/ybinWidth)

    # Peform the kernel density estimate
    #xx, yy = np.mgrid[xmin:xmax:xbin*1j, ymin:ymax:ybin*1j]
    #positions = np.vstack([xx.ravel(), yy.ravel()])
    #values = np.vstack([x, y])

    xbin_centers, ybin_centers, kde_2d = vonmises_fft_sphere_kde(
        x, 10, xbin, xmin, xmax,
        y, 10, ybin, ymin, ymax)

    #kernel = ss.gaussian_kde(values)
    #f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(7, 6))
    grid = plt.GridSpec(4, 5, hspace=0.5, wspace=0.5)
    main_ax = fig.add_subplot(grid[1:, :-2])
    y_plot = fig.add_subplot(grid[1:, -2], xticklabels=[], sharey=main_ax)
    x_plot = fig.add_subplot(grid[0, :-2], yticklabels=[], sharex=main_ax)
    cbaxes = fig.add_subplot(grid[1:, -1])

    main_ax.set_xlim(xmin, xmax)
    main_ax.set_ylim(ymin, ymax)
    # Contourf plot

    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    ti = 0
    tj = 0
    max_val = max(map(max, kde_2d))
    for col in kde_2d:
        for val in col:
            val = to_energy(val, max_val)
            kde_2d[ti][tj] = val
            tj += 1
        ti += 1
        tj = 0
    CS = main_ax.contourf(xbin_centers, ybin_centers, kde_2d.T, 50, cmap=cmapName)
    cb = fig.colorbar(CS, cax=cbaxes)
    cb.set_label("Density")
    # histogram on the attached axes
    xbin_centers, xkde = vonmises_fft_kde(x, 10, xbin, xmin, xmax)
    x_plot.plot(xbin_centers, xkde)

    ybin_centers, ykde = scipy_kde(y, ybin, ymin, ymax)
    y_plot.plot(ykde, ybin_centers)
    main_ax.set_xlabel(xname)
    main_ax.set_ylabel(yname)
    fig.suptitle(title, fontsize=16)
    fig.savefig(name, format='pdf')
    
def Flat_KDE(name, title, xname, yname, x, xmin, xmax, y, ymin, ymax, cmapName, proportion):
    xbinWidth = bw_scott(x)*proportion
    ybinWidth = bw_scott(y)*proportion

    xbin = round_up_to_even((xmax - xmin)/xbinWidth)
    ybin = round_up_to_even((ymax - ymin)/ybinWidth)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:xbin*1j, ymin:ymax:ybin*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    kernel = ss.gaussian_kde(values)
    kde_2d = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(7, 6))
    grid = plt.GridSpec(4, 5, hspace=0.5, wspace=0.5)
    main_ax = fig.add_subplot(grid[1:, :-2])
    y_plot = fig.add_subplot(grid[1:, -2], xticklabels=[], sharey=main_ax)
    x_plot = fig.add_subplot(grid[0, :-2], yticklabels=[], sharex=main_ax)
    cbaxes = fig.add_subplot(grid[1:, -1])

    main_ax.set_xlim(xmin, xmax)
    main_ax.set_ylim(ymin, ymax)
    # Contourf plot

    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    ti = 0
    tj = 0
    max_val = max(map(max, kde_2d))
    for col in kde_2d:
        for val in col:
            val = to_energy(val, max_val)
            kde_2d[ti][tj] = val
            tj += 1
        ti += 1
        tj = 0
    CS = main_ax.contourf(xx, yy, kde_2d, 50, cmap=cmapName)
    cb = fig.colorbar(CS, cax=cbaxes)
    cb.set_label("Density")
    # histogram on the attached axes
    xbin_centers, xkde = scipy_kde(x, xbin, xmin, xmax)
    x_plot.plot(xbin_centers, xkde)

    ybin_centers, ykde = scipy_kde(y, ybin, ymin, ymax)
    y_plot.plot(ykde, ybin_centers)
    main_ax.set_xlabel(xname)
    main_ax.set_ylabel(yname)
    fig.suptitle(title, fontsize=16)
    fig.savefig(name, format='pdf')

def length(v):
    return math.sqrt(np.dot(v, v))

def unit(v):
    return v / length(v)

def angle_calc(v1, v2):
    return math.acos(round(np.dot(v1, v2) / (length(v1) * length(v2)), 3))

def angleWithPlane(v1, v2):
    return math.asin(np.dot(v1, v2) / (length(v1) * length(v2)))

def dihedral(v1, v2, vn):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    # normalize vn so that it does not influence magnitude of vector
    # projections that come next
    vn = unit(vn)

    # vector rejections
    # v = projection of v1 onto plane perpendicular to vn
    #   = v1 minus component that aligns with vn
    # w = projection of v2 onto plane perpendicular to vn
    #   = v2 minus component that aligns with vn
    v = v1 - np.dot(v1, vn)*vn
    w = v2 - np.dot(v2, vn)*vn

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    dot = np.dot(v, w)
    a = np.array([v, w, vn])
    det = np.linalg.det(a)
    torsion = math.atan2(det, dot)
    return torsion

def newCord(o, t, a):
    value1 = t[0]*(t[0]*o[0]+t[1]*o[1]+t[2]*o[2])*(1-math.cos(a))+o[0]*math.cos(a)+(-t[2]*o[1]+t[1]*o[2])*math.sin(a)
    value2 = t[1]*(t[0]*o[0]+t[1]*o[1]+t[2]*o[2])*(1-math.cos(a))+o[1]*math.cos(a)+(t[2]*o[0]-t[0]*o[2])*math.sin(a)
    value3 = t[2]*(t[0]*o[0]+t[1]*o[1]+t[2]*o[2])*(1-math.cos(a))+o[2]*math.cos(a)+(-t[1]*o[0]+t[0]*o[1])*math.sin(a)
    return np.array([value1, value2, value3])

def rotateX(o, a):
    return np.dot(np.array([[1, 0, 0, 0], [0, math.cos(a), -math.sin(a), 0],
                            [0, math.sin(a), math.cos(a), 0],
                            [0, 0, 0, 1]]), np.array([o[0], o[1], o[2], 1]))

def readAtoms(line, start, size):
    output = [None for _ in range(4)]
    output[0] = line[11:21]
    temp = list(filter(None, line[31:-25].split(" ")))
    output[1] = round(float(temp[0]), 3)
    output[2] = round(float(temp[1]), 3)
    output[3] = round(float(temp[2]), 3)
    axis = 1
    while axis <= 3:
        if abs(output[axis] - start[axis-1]) > abs(output[axis] - size[axis-1] - start[axis-1]):
            output[axis] = output[axis] - size[axis-1]
        elif abs(output[axis] - start[axis-1]) > abs(output[axis] + size[axis-1] - start[axis-1]):
            output[axis] = output[axis] + size[axis-1]
        axis += 1
    return output

def readMolecule(inputFile, size, atoms, atomset):
    count = atoms+1
    output = [[None for _ in range(4)] for _ in range(count)]
    a = 0
    start = [None for _ in range(3)]
    line = inputFile.readline()
    output[a][0] = line[11:21]
    temp = list(filter(None, line[31:-25].split(" ")))
    start[0] = output[a][1] = round(float(temp[0]), 3)
    start[1] = output[a][2] = round(float(temp[1]), 3)
    start[2] = output[a][3] = round(float(temp[2]), 3)
    a += 1
    while a < atoms:
        line = inputFile.readline()
        output[a] = readAtoms(line, start, size)
        a += 1
    center = getAveragePosition(output, atomset)
    output[a] = ["center", center[0], center[1], center[2]]
    return output

def calcPlaneVector(points, centroid, n):
    # Calculate full 3x3 covariance matrix, excluding symmetries:
    xx = 0.0; xy = 0.0; xz = 0.0
    yy = 0.0; yz = 0.0; zz = 0.0

    for p in points:
        r = np.subtract(p, centroid)
        xx += r[0] * r[0]
        xy += r[0] * r[1]
        xz += r[0] * r[2]
        yy += r[1] * r[1]
        yz += r[1] * r[2]
        zz += r[2] * r[2]

    xx /= n
    xy /= n
    xz /= n
    yy /= n
    yz /= n
    zz /= n

    weighted_dir = np.array([0.0, 0.0, 0.0])

    det_x = yy*zz - yz*yz
    axis_dir = np.array([det_x, xz*yz - xy*zz, xy*yz - xz*yy])
    weight = det_x * det_x
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight = -weight
    weighted_dir += axis_dir * weight

    det_y = xx*zz - xz*xz
    axis_dir = np.array([xz*yz - xy*zz, det_y, xy*xz - yz*xx])
    weight = det_y * det_y
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight = -weight
    weighted_dir += axis_dir * weight

    det_z = xx*yy - xy*xy
    axis_dir = np.array([xy*yz - xz*yy, xy*xz - yz*xx, det_z])
    weight = det_z * det_z
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight = -weight
    weighted_dir += axis_dir * weight

    normal = unit(weighted_dir)

    inv = points[1] # Any C atom that is not in a straigh line with the centare and carboxil group

    v1 = points[0] - centroid # The C atom connected to the carboxil group
    v2 = inv - centroid
    tors = dihedral(v1, v2, normal)
    if tors < 0:
        normal = -normal
    return normal

def selectPlanesAtoms(molecule, atomSet):
    points = []

    for atom in atomSet:
        point = np.array([molecule[atom][1], molecule[atom][2], molecule[atom][3]])
        points.append(point)
    n = len(points)

    total = np.array([0, 0, 0])
    for p in points:
        total = np.add(total, p)

    centroid = np.true_divide(total, n)
    return points, centroid, n

def calcTransfer(normal, points, centroid):
    # A vecotor that defines XY plane
    targetPlane = np.array([0, 0, 1])
    # Cross product of existing plane vecotr and target plane vector
    planeCross = unit(np.cross(normal, targetPlane))
    # Angle between existing plane vector and target plane vector
    planeAngle = angle_calc(targetPlane, normal)
    temp = np.array([points[0][0] - centroid[0],
                     points[0][1] - centroid[1],
                     points[0][2] - centroid[2]])
    temp = newCord(temp, planeCross, planeAngle)
    dB = temp[2]
    temp = np.array([temp[0], temp[1], 0])
    # The acid groupd carbon atom target location
    targetA = np.array([(temp[0]**2+temp[1]**2)**0.5, 0, temp[2]])
    # Cross product of target A and existing A
    crossA = unit(np.cross(temp, targetA))
    # Angle between existing A and target A
    angleA = angle_calc(targetA, temp)
    return planeCross, crossA, planeAngle, angleA, dB

def getMoleculeOrientationX(vn, centroid, orienterAtom):
    # conversion from cartesian to polar coordinates
    torsion = math.atan2(vn[1], vn[0])
    # conversion from cartesian to polar coordinates
    thisAngle = math.acos(vn[2])
    v1 = np.array([orienterAtom[0] - centroid[0],
                   orienterAtom[1] - centroid[1],
                   orienterAtom[2] - centroid[2]])
    v2 = np.array([1, 0, 0])
    rotation = dihedral(v1, v2, vn)
    relAngle = angle_calc(np.array([0.0, 0.0, 1.0]), vn)
    #if(relAngle < 0):
    #    relAngle = np.pi + relAngle;
    #if(relAngle > np.pi/2):
    #    relAngle = np.pi - relAngle;
    line = np.array([-centroid[0], -centroid[1], -centroid[2]])
    # calculate angle between the carboxil group and the line connceting the molecules
    # (small angle facing the central molecule, large angle facing away from central molecule)
    direction = angle_calc(v1, line)
    if direction < 0:
        direction = np.pi + direction
    return thisAngle, torsion, rotation, direction, relAngle

def recalAsHexColor(angle, torsion, rotation):
    r = int(angle/np.pi*255)
    g = int((torsion+np.pi)/(2*np.pi)*255)
    b = int((rotation+np.pi)/(2*np.pi)*255)
    hexCode = '#%02x%02x%02x' % (r, g, b)
    return hexCode

def moveMol(mol, rx, ry, rz, size):
    for atom in mol:
        if rx == -1:
            atom[1] = atom[1]-size[0]
        elif rx == 1:
            atom[1] = atom[1]+size[0]
        if ry == -1:
            atom[2] = atom[2]-size[1]
        elif ry == 1:
            atom[2] = atom[2]+size[1]
        if rz == -1:
            atom[3] = atom[3]-size[2]
        elif rz == 1:
            atom[3] = atom[3]+size[2]
    return mol

def calcDistanceSimple(a1, a2):
    dx = np.abs(a1[1] - a2[1])
    dy = np.abs(a1[2] - a2[2])
    dz = np.abs(a1[3] - a2[3])
    return (dx**2 + dy**2 + dz**2)**0.5

def calcDistance(a1, a2, boxSize):
    rx = 0
    ry = 0
    rz = 0
    dx = a1[1] - a2[1]
    if dx > boxSize[0]/2:
        dx = boxSize[0] - dx
        rx = 1
    elif dx < -boxSize[0]/2:
        dx = boxSize[0] + dx
        rx = -1
    dy = a1[2] - a2[2]
    if dy > boxSize[1]/2:
        dy = boxSize[1] - dy
        ry = 1
    elif dy < -boxSize[1]/2:
        dy = boxSize[1] + dy
        ry = -1
    dz = a1[3] - a2[3]
    if dz > boxSize[2]/2:
        dz = boxSize[2] - dz
        rz = 1
    elif dz < -boxSize[2]/2:
        dz = boxSize[2] + dz
        rz = -1

    return (dx**2 + dy**2 + dz**2)**0.5, rx, ry, rz

def calcMolVector(mol1, mol2, atomSet1, atomSet2):
    coord1 = getAveragePosition(mol1, atomSet1)
    coord2 = getAveragePosition(mol2, atomSet2)
    vector = coord1 - coord2
    return vector

#def calMolDisctance(mol1, mol2, atomSet):
#    vector = calcMolVector(mol1, mol2, atomSet);
#    return length(vector);

def checkHydrogenBond(O1, O2, H):
    dis = calcDistanceSimple(O1, O2)
    v1 = [O1[1] - H[1], O1[2] - H[2], O1[3] - H[3]]
    v2 = [O2[1] - H[1], O2[2] - H[2], O2[3] - H[3]]
    ang = angle_calc(v1, v2)
    loose = (dis <= 4.0 and ang > 1.744)
    tight = (dis <= 3.2 and ang > 2.268)
    return tight, loose, dis, ang

def addToList(theList, identifier1, identifier2):
    found = False
    for element in theList:
        if ((element[0][0] == identifier1) and (element[0][1] == identifier2)):
            found = True
            element[0] += 1
            break
    if not found:
        theList.append(np.array([[identifier1, identifier2, 0]]))
    return theList

def checkHydrogenBonds(donorsA, donorsB, acceptorsA, acceptorsB, molA, molB):
    HBondsTight = []
    HBondsLoose = []
    isHBondTight = False
    isHBondLoose = False
    descriptors = []
    distances = []

    for acceptor in acceptorsB:
        for donor in donorsA:
            if not donor or not acceptor:
                break
            dis2 = calcDistanceSimple(molA[donor[0]], molB[acceptor])
            distance = ["D", donor[0], dis2]
            distances.append(distance)
            for hydrogen in donor[1]:
                t1, l1, dis, ang = checkHydrogenBond(molA[donor[0]], molB[acceptor], molA[hydrogen])
                descriptor = {"Primary":{"Type": "D", "Donor": donor[0], "Hydrogen": hydrogen},
                              "Secondary":{"Type": "A", "Acceptor": acceptor},
                              "Distance": dis, "Angle": ang, "TightBond": t1, "LooseBond": l1}
                descriptors.append(descriptor)
                if l1:
                    HBondsLoose = addToList(HBondsLoose, "D", donor[0])
                    isHBondLoose = True
                    if t1:
                        HBondsTight = addToList(HBondsTight, "D", donor[0])
                        isHBondTight = True
       
    for acceptor in acceptorsA:
        for donor in donorsB:
            if not donor or not acceptor:
                break
            dis2 = calcDistanceSimple(molB[donor[0]], molA[acceptor])
            distance = ["A", acceptor, dis2]
            distances.append(distance)
            for hydrogen in donor[1]:
                t1, l1, dis, ang = checkHydrogenBond(molB[donor[0]], molA[acceptor], molB[hydrogen])
                descriptor = {"Secondary":{"Type": "D", "Donor": donor[0], "Hydrogen": hydrogen},
                              "Primary":{"Type": "A", "Acceptor": acceptor},
                              "Distance": dis, "Angle": ang, "TightBond": t1, "LooseBond": l1}
                descriptors.append(descriptor)
                if l1:
                    HBondsLoose = addToList(HBondsLoose, "A", acceptor)
                    isHBondLoose = True
                    if t1:
                        HBondsTight = addToList(HBondsTight, "A", acceptor)
                        isHBondTight = True

    return isHBondTight, isHBondLoose, HBondsTight, HBondsLoose, descriptors, distances

def checkPiPiStacking(molA, nA, molB, nB, atomSet, dist):
    tight = False
    vec = calcMolVector(molA, molB, atomSet, atomSet)
    if dist >= 6.0:
        return False, False, "Null"
    if dist <= 5.6:
        tight = True
    gamma = angle_calc(nA, nB) # angle between planes
    if gamma < 0:
        gamma = -gamma
    if gamma > math.pi/2:
        gamma = math.pi - gamma
    theta = angleWithPlane(vec, nA) # angle between 1. plane and vector that conncets the molecules
    if theta < 0:
        theta = -theta
    if theta > math.pi/2:
        theta = math.pi - theta
    delta = angleWithPlane(vec, nB) # angle between 2. plane and vector that conncets the molecules\
    if delta < 0:
        delta = -delta
    if delta > math.pi/2:
        delta = math.pi - delta
    if theta < math.pi/6 and delta < math.pi/6:
        return False, False, "Null"
    if gamma > math.pi*5/18: #if grater than 50 degrees
        return tight, True, "Tshape" # T-shape conformation
    if gamma >= math.pi/6:
        return tight, True, "Intermedian" # intemedian conformation
    if gamma < math.pi/6:
        if theta > math.pi*8/18: #if grater than 80 degrees
            return tight, True, "FtF" # face to face conformation
        return tight, True, "Offset" # Offset conformation

def getAveragePosition(molecule, atomSet):
    coordinates = np.array([0.0, 0.0, 0.0])
    i = 0
    while i < len(atomSet):
        atom = atomSet[i]
        coordinates[0] = (coordinates[0] * i + molecule[atom][1])/(i+1)
        coordinates[1] = (coordinates[1] * i + molecule[atom][2])/(i+1)
        coordinates[2] = (coordinates[2] * i + molecule[atom][3])/(i+1)
        i += 1
    return coordinates

def position4to3(molecule, atomNr):
    return np.array([molecule[atomNr][1], molecule[atomNr][2], molecule[atomNr][3]])

def getPolarCoordinatesZX(targetMol, centralAtom):
    v1 = np.array([0.0, 0.0, 1.0]) # z-axis unit vector
    v2 = position4to3(targetMol, centralAtom)
    Zang = angle_calc(v1, v2)
    vd = np.array([1.0, 0.0, 0.0])
    Ztors = dihedral(vd, v2, v1)
    return Zang, Ztors

def writePDB(moleculeA, moleculeB, size):
    output = []
    output.append("REMARK    GENERATED BY TRJCONV\n")
    output.append("REMARK    THIS IS A SIMULATION BOX\n")
    output.append("CRYST1  "+str(size[0])+"  "+str(size[1])+"  "+str(size[2])+
                  "  90.00  90.00  90.00 P 1           1\n")

    atomNr = 1
    for atom in moleculeA:
        if atom[0] == "center":
            continue
        AtNumber = str(atomNr).rjust(3)
        MolNumber = str(1).rjust(5)
        x = str(round(atom[1], 3)).rjust(7)
        y = str(round(atom[2], 3)).rjust(7)
        z = str(round(atom[3], 3)).rjust(7)
        output.append("ATOM    " + AtNumber + atom[0] + MolNumber +
                      "     " + x + " " + y + " " + z + "  1.00  0.00            \n")
        atomNr += 1
    for atom in moleculeB:
        if atom[0] == "center":
            continue
        AtNumber = str(atomNr).rjust(3)
        MolNumber = str(2).rjust(5)
        x = str(round(atom[1], 3)).rjust(7)
        y = str(round(atom[2], 3)).rjust(7)
        z = str(round(atom[3], 3)).rjust(7)
        output.append("ATOM    " + AtNumber + atom[0] + MolNumber +
                      "     " + x + " " + y + " " + z + "  1.00  0.00            \n")
        atomNr += 1

    output.append("TER\n")
    output.append("ENDMDL\n")
    return output

def transferMolecule(molecule, centroid, planeCross, crossA, planeAngle, angleA, dB):
    newMol = [[None for _ in range(4)] for _ in range(len(molecule))]
    i = 0
    for atom in molecule:
        temp = np.array([atom[1] - centroid[0],
                         atom[2] - centroid[1],
                         atom[3] - centroid[2]])
        temp = newCord(temp, planeCross, planeAngle)
        temp = np.array([temp[0], temp[1], temp[2]-dB])
        temp = newCord(temp, crossA, angleA)
        nCord = np.array([temp[0], temp[1], temp[2]+dB])
        newMol[i] = [atom[0], nCord[0], nCord[1], nCord[2]]
        i += 1
    return newMol

def take_closest(myList, myNumber, lower):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the bigger number.
    """
    if lower:
        return bisect_left(myList, myNumber)
    return bisect_right(myList, myNumber)

def addToCluster(clusterList, dimer):
    f1 = False
    f2 = False
    i = 0
    e1 = 0
    e2 = 0
    for entity in clusterList:
        if dimer[0] in entity:
            e1 = i
            f1 = True
        if dimer[1] in entity:
            e2 = i
            f2 = True
        i += 1
    if not f1 and not f2:
        clusterList.append([dimer[0], dimer[1]])
    elif f1 and f2:
        if clusterList[e1] != clusterList[e2]:
            for element in clusterList[e2]:
                clusterList[e1].append(element)
            clusterList.remove(clusterList[e2])
    elif f1 and not f2:
        clusterList[e1].append(dimer[1])
    elif f2 and not f1:
        clusterList[e2].append(dimer[0])
    return clusterList

def calInternalAngle(mol, v1atoms, v2atoms):
    v1 = calcMolVector(mol, mol, [v1atoms[0]], [v1atoms[1]])
    v2 = calcMolVector(mol, mol, [v2atoms[0]], [v2atoms[1]])
    interAngle = angle_calc(v1, v2)
    return interAngle

def calInternalTorsion(mol, v1atoms, v2atoms, vnatoms):
    v1 = calcMolVector(mol, mol, [v1atoms[0]], [v1atoms[1]])
    v2 = calcMolVector(mol, mol, [v2atoms[0]], [v2atoms[1]])
    vn = calcMolVector(mol, mol, [vnatoms[0]], [vnatoms[1]])
    interTorsion = dihedral(v1, v2, vn)
    return interTorsion

def exportPDB(filenames, inPath, outPath):
    with open(outPath, 'w') as outfile:
        for fname in filenames:
            with open(os.path.join(inPath, fname)) as infile:
                outfile.write(infile.read())

def saveText(text, outputFile):
    print(text)
    with open(outputFile, 'a') as the_file:
        the_file.write(text+"\n")

def processFile(molecules, mol_atoms, solvent, sol_atoms,
                path, frame_nr, output_location,
                mol_center_atoms, sol_center_atoms, mol_plane_atoms,
                check_hydrogen_intermolecular, mol_donors, mol_acceptors,
                check_hydrogen_withsolvent, sol_donors, sol_acceptors,
                check_internal, internal_rotation, internal_torsion,
                check_pipi_intermolecular, mol_benzol, now, symetryAtoms):
    molCentralAtom = mol_atoms
    solCentralAtom = sol_atoms

    with open(path, "r") as input_file:
        old_system = [[[None for _ in range(4)] for _ in range(mol_atoms)]
                      for _ in range(molecules)]
        sol_system = [[[None for _ in range(4)] for _ in range(sol_atoms)]
                      for _ in range(solvent)]

        #############################
        ## Reading in single frame ##
        #############################

        # Chekcking if first line exists 'REMARK    GENERATED BY TRJCONV`
        line = input_file.readline()
        if not line:
            return
        input_file.readline() # skipping 'TITLE     2FB in water t=   0.00000 step= 0'
        input_file.readline() # skipping 'REMARK    THIS IS A SIMULATION BOX'
        # Ŗeading box size 'CRYST1  142.182  142.182  142.182  90.00  90.00  90.00 P 1           1'
        line = input_file.readline();
        sizeX = round(float((line)[8:15]), 3)
        sizeY = round(float((line)[17:24]), 3)
        sizeZ = round(float((line)[26:33]), 3)
        
        size = [sizeX, sizeY, sizeZ]
        
        frameNumber = int((input_file.readline())[5:]) # skipping 'MODEL        1'

        # Reading molecules in
        m = 0
        while m < molecules:
            old_system[m] = readMolecule(input_file, size, mol_atoms, mol_center_atoms)
            m += 1

        # Reading solvents in
        m = 0
        while m < solvent:
            sol_system[m] = readMolecule(input_file, size, sol_atoms, sol_center_atoms)
            m += 1
        input_file.readline() # Skipping 'TER'
        input_file.readline() # Skipping 'ENDMDL'

        #####################################
        ## Processing frames molecule data ##
        #####################################
        moleculeData = []

        relativeOrientation = []
        sphereData = []
        clusterSphereData = []
        piAverage = 0
        HAverage = 0
        piTypes = []
        DesityRadius = 10
        pipiCounter = {
            "pipiF": 0,
            "pipiT": 0,
            "pipiO": 0,
            "pipiI": 0
        }
        rotation = []
        directions = []

        HBondDescriptor = []
        HBondsSolvent = []
        Sym = 0
        Asym = 0
        Density = []

        outputMol = []
        outputSol = []
        clusters = []
        looseClusters = []
        internalAngles = []
        i = 0

        while i < molecules: # For each main molecule in frame do...
            ############################################
            ## Calculating transformation paramathers ##
            ############################################
            points, centroid, n = selectPlanesAtoms(old_system[i], mol_plane_atoms)
            normalBase = calcPlaneVector(points, centroid, n)
            planeCross, crossA, planeAngle, angleA, dB = calcTransfer(
                    normalBase, points, centroid)

            if check_internal:
                intvec1 = internal_rotation[0]
                intvec2 = internal_rotation[1]
                internalAngle = calInternalAngle(old_system[i], intvec1, intvec2)
                intvec1 = internal_torsion[0]
                intvec2 = internal_torsion[1]
                intvec3 = internal_torsion[2]
                internalTorsion = calInternalTorsion(old_system[i], intvec1, intvec2, intvec3)
                internalAngles.append({"internalAngle":internalAngle,
                                       "internalTorsion":internalTorsion,
                                       "molecule":i, "frame":frameNumber})
                internalAngles.append({"internalAngle":2*np.pi-internalAngle,
                                       "internalTorsion":internalTorsion,
                                       "molecule":i, "frame":frameNumber})

            ##############################################
            ### Calcaulte intercation between moleculs ###
            ##############################################

            if check_hydrogen_intermolecular or check_pipi_intermolecular:
                j = 0
                while j < molecules: # To do calculation with other main molecules
                    if i == j:
                        j += 1
                        continue
                    # If distance between molecule centers is more than 8 ignore
                    dis, rx, ry, rz = calcDistance(old_system[i][molCentralAtom],
                                                   old_system[j][molCentralAtom],
                                                   size)
                    if dis > 11:
                        j += 1
                        continue

                    tempMol = moveMol(old_system[j], rx, ry, rz, size)
                    # trasnfer/roate target and adjacent molecules
                    molA = transferMolecule(old_system[i], centroid, planeCross,
                                            crossA, planeAngle, angleA, dB)
                    molB = transferMolecule(tempMol, centroid, planeCross,
                                            crossA, planeAngle, angleA, dB)

                    #calculate other molecules postition in space in respct to the first one
                    Zang, Ztors = getPolarCoordinatesZX(molB, molCentralAtom)

                    #calculate orientation of the end molecule in polar coordinates
                    points, newCentroid, n = selectPlanesAtoms(molB, mol_plane_atoms)
                    normalNew = calcPlaneVector(points, newCentroid, n)
                    # The atom we use to orientate our molecule
                    orienterAtom = points[0]
                    molAng, molTors, molRot, direction, relAngle = getMoleculeOrientationX(
                        normalNew, newCentroid, orienterAtom)
                    #check if there is pi-pi stacking
                    isPiPiTight, isPiPiLoose, piType = checkPiPiStacking(molA,
                                                                         np.array([0.0, 0.0, 1.0]),
                                                                         molB, normalNew,
                                                                         mol_benzol, dis)

                    # Check if HBond exists and with what atom
                    isHBondTight, isHBondLoose, HBondsTight, HBondsLoose, descriptors, distances = checkHydrogenBonds(
                        mol_donors, mol_donors, mol_acceptors, mol_acceptors, molA, molB)
                    if symetryAtoms:
                        symetryCond = [[symetryAtoms[0], False], [symetryAtoms[1], False]]               
                        for descriptor in descriptors:
                            HBtype = descriptor["Primary"]["Type"]
                            atom = -1
                            if HBtype == "D":
                                atom = descriptor["Primary"]["Donor"]
                            else:
                                atom = descriptor["Primary"]["Acceptor"]
                            name = HBtype + str(atom)
                            if descriptor["TightBond"] == True:
                                elem = next((x for x in symetryCond if x[0] == name), None)
                                if elem != None:
                                    elem[1] = True
                        bondExists = any(elem[1] == True for elem in symetryCond)

                        if bondExists:
                            simetricalBond = not any(elem[1] == False for elem in symetryCond)
                            if simetricalBond:
                                Sym += 1
                            else:
                                Asym += 1
                    
                    HBondSets = []
                    for descriptor in descriptors:
                        if descriptor["TightBond"]:
                            primaryAtom = ""
                            secondaryAtom = ""

                            if descriptor["Primary"]["Type"] == "D":
                                primaryAtom = "D" + str(descriptor["Primary"]["Donor"])
                                secondaryAtom = "A" + str(descriptor["Secondary"]["Acceptor"])
                            else:
                                primaryAtom = "A" + str(descriptor["Primary"]["Acceptor"])
                                secondaryAtom = "D" + str(descriptor["Secondary"]["Donor"])

                            HBondSets.append([primaryAtom, secondaryAtom])

                    molData = {"pairName":str(i)+"-"+str(j),
                               "isPiPiTight":isPiPiTight,
                               "isPiPiLoose":isPiPiLoose,
                               "pipiType":piType,
                               "isHBondTight":isHBondTight,
                               "isHBondLoose":isHBondLoose,
                               "HBondSets": HBondSets}
                    moleculeData.append(molData)

                    relativeOrientation.append({"distance": dis,
                                                "orientation": relAngle})
                    relativeOrientation.append({"distance": dis,
                                                "orientation": -relAngle})

                    AdjacentAsColor = recalAsHexColor(molAng, molTors, molRot)
                    sphereData.append({"radius": dis, "CentralPolarAngle": Zang,
                                       "CentralPolarTorsion":Ztors,
                                       "AdjacentPolar":AdjacentAsColor})
                        
                    # Checking PiPi interaction between molecules
                    if isPiPiTight:
                        piAverage += 1/molecules
                        result = next((elem for elem in piTypes if elem[0] == piType), None)
                        u = math.floor(dis/0.1)
                        if result == None:
                            elem = [piType, [[k*0.1, 0] for k in range(DesityRadius*10)]]
                            result = elem
                            piTypes.append(elem)

                        index = piTypes.index(result)
                        if dis < DesityRadius:
                            piTypes[index][1][u][1] += 1/(4/3*np.pi*((((u+1)*0.1)**3)-((u*0.1)**3)))

                        if piType in ('Offset', 'FtF'):
                            if piType == 'Offset':
                                pipiCounter["pipiO"] += 1
                            if piType == 'FtF':
                                pipiCounter["pipiF"] += 1
                            rotation.append(molRot)
                        if piType == 'Tshape':
                            pipiCounter["pipiT"] += 1
                            directions.append(direction)
                        if piType == 'Intermedian':
                            pipiCounter["pipiI"] += 1

                    if HBondsTight:
                        HAverage += 1/molecules

                    pair = [i, j]

                    if(not isHBondLoose and not isPiPiLoose):
                        j += 1
                        continue

                    looseClusters = addToCluster(looseClusters, pair)

                    if(not isHBondTight and not isPiPiTight):
                        j += 1
                        continue

                    clusters = addToCluster(clusters, pair)

                    clusterSphereData.append({"radius": dis,
                                              "CentralPolarAngle": Zang,
                                              "CentralPolarTorsion":Ztors,
                                              "AdjacentPolar":AdjacentAsColor})

                    # write dimer to file
                    retMol = writePDB(molA, molB, size)
                    outputMol.extend(retMol)
                    j += 1

            ##########################################################
            ### Calcaulte intercation between molecule and solvent ###
            ##########################################################

            center = old_system[i][molCentralAtom]
            sol_system.sort(key=lambda x: x[solCentralAtom][1]) # Sort solvent molecules by oxigen atom x-coord
            x_sol_keys = [r[solCentralAtom][1] for r in sol_system]
            j = take_closest(x_sol_keys, center[1]-11, True)
            e = take_closest(x_sol_keys, center[1]+11, False)

            temp_sol = sol_system[j:e]
            temp_sol.sort(key=lambda x: x[solCentralAtom][2]) # Sort solvent molecules by oxigen atom y-coord
            y_sol_keys = [r[solCentralAtom][2] for r in temp_sol]
            j = take_closest(y_sol_keys, center[2]-11, True)
            e = take_closest(y_sol_keys, center[2]+11, False)

            temp_sol = temp_sol[j:e]
            temp_sol.sort(key=lambda x: x[solCentralAtom][3]) # Sort solvent molecules by oxigen atom z-coord
            z_sol_keys = [r[solCentralAtom][3] for r in temp_sol]
            j = take_closest(z_sol_keys, center[3]-11, True)
            e = take_closest(z_sol_keys, center[3]+11, False)
            while j < e: # To do calculation with solvent molecules

                # trasnfer/roate target and adjacent molecules
                molA = transferMolecule(old_system[i], centroid,
                                        planeCross, crossA,
                                        planeAngle, angleA, dB)
                molB = transferMolecule(temp_sol[j], centroid,
                                        planeCross, crossA,
                                        planeAngle, angleA, dB)

                # Check if HBond exists and with what atom
                isHBondTight, isHBondLoose, HBondsTight, HBondsLoose, descriptors, distances = checkHydrogenBonds(
                    mol_donors, sol_donors, mol_acceptors, sol_acceptors, molA, molB)

              
                for descriptor in descriptors:
                    HBtype = descriptor["Primary"]["Type"]
                    atom = -1
                    if HBtype == "D":
                        atom = descriptor["Primary"]["Donor"]
                    else:
                        atom = descriptor["Primary"]["Acceptor"]
                    name = HBtype + str(atom)
                    if descriptor["TightBond"] == True:
                        dist = descriptor["Distance"]
                        ang = descriptor["Angle"]
                        result = next((elem for elem in HBondsSolvent if elem[0] == name), None)
                        if result == None:
                            HBondsSolvent.append([name, [[dist, ang]]])
                        else:
                            result[1].append([dist, ang])

                for descriptor in descriptors:
                    destriptorType = descriptor["Primary"]["Type"]
                    descriptorName = ""
                    if destriptorType == "D":
                        descriptorName = destriptorType + str(descriptor["Primary"]["Donor"])
                    else:
                        descriptorName = destriptorType + str(descriptor["Primary"]["Acceptor"])
                    if descriptor["TightBond"]:
                        result = next((elem for elem in HBondDescriptor if
                                       elem[0] == descriptorName),
                                      None)
                        if result == None:
                            HBondDescriptor.append([descriptorName, 1])
                        else:
                            index = HBondDescriptor.index(result)
                            HBondDescriptor[index][1] += 1

                for distanceData in distances:
                    HBtype = distanceData[0]
                    atom = distanceData[1]
                    result = next((elem for elem in Density if elem[0] == atom),
                                  None)
                    result2 = next((elem for elem in Density if
                                    (elem[0] == atom and elem[1] == HBtype)),
                                   None)
                    if result == None:
                        elem = [atom, HBtype, [[k*0.1, 0] for k in range(8*10)]]
                        Density.append(elem)
                        result = elem
                    elif result2 == None:
                        continue

                    dist = distanceData[2]
                    index = Density.index(result)
                    u = math.floor(dist/0.1)
                    if dist < 8:
                        Density[index][2][u][1] += 1/(4/3*np.pi*((((u+1)*0.1)**3)-((u*0.1)**3)))

                if not isHBondTight:
                    j += 1
                    continue

                # write dimer to file
                retSol = writePDB(molA, molB, size)
                outputSol.extend(retSol)
                j += 1
            i += 1

        with open(os.path.join(output_location, "temp-"+now,
                               "temp"+str(frameNumber)+"_mol.pdb"), "w") as of:
            of.writelines(outputMol)
        with open(os.path.join(output_location, "temp-"+now,
                               "temp"+str(frameNumber)+"_sol.pdb"), "w") as of:
            of.writelines(outputSol)
    with open(os.path.join(output_location, "temp-"+now,
                           'temp'+str(frameNumber)+'.pkl'), 'wb') as f:
        pickle.dump([HBondDescriptor, moleculeData, clusters,
                     looseClusters, internalAngles, relativeOrientation,
                     sphereData, piAverage, piTypes, pipiCounter,
                     rotation, directions, HAverage, HBondsSolvent,
                     Sym, Asym, Density, clusterSphereData], f)
    return HBondDescriptor, moleculeData, clusters, looseClusters, internalAngles, relativeOrientation, sphereData, piAverage, piTypes, pipiCounter, rotation, directions, HAverage, HBondsSolvent, Sym, Asym, Density, clusterSphereData

def workerNew(input_data):
    try:
        pkl = os.path.join(input_data[6],
                           "temp-"+input_data[21],
                           "temp"+str(input_data[5])+".pkl")
        mol_pdb = os.path.join(input_data[6],
                               "temp-"+input_data[21],
                               "temp"+str(input_data[5])+"_mol.pdb")
        sol_pdb = os.path.join(input_data[6],
                               "temp-"+input_data[21],
                               "temp"+str(input_data[5])+"_sol.pdb")
        if not os.path.isfile(pkl) or not os.path.isfile(mol_pdb) or not os.path.isfile(sol_pdb):
            print("doing frame - " + str(input_data[5]))
            HBondDescriptor, moleculeData, clusters, looseClusters, internalAngles, relativeOrientation, sphereData, piAverage, piTypes, pipiCounter, rotation, directions, HAverage, HBondsSolvent, Sym, Asym, Density, clusterSphereData = processFile(
                input_data[0],  # molecules
                input_data[1],  # mol_atoms
                input_data[2],  # solvent
                input_data[3],  # sol_atoms
                input_data[4],  # temp1.temp
                input_data[5],  # frame nr
                input_data[6],  # output folder
                input_data[7],  # molecule center defining atoms
                input_data[8],  # solvent center defining atoms
                input_data[9],  # molecule plane defining atoms
                input_data[10], # boolean - do we check intermolecular hydrogen bond interactions
                input_data[11], # molecule donor atoms with their hydrogens
                input_data[12], # molecule acceptor atoms
                input_data[13], # boolean - do we check hydrogen bond interactions with solvent
                input_data[14], # solvent donor atoms with their hydrogens
                input_data[15], # solvent acceptors
                input_data[16], # boolean - do we check internal molecule rotations
                input_data[17], # interanl rotation vectors
                input_data[18], # internal torsion vectors
                input_data[19], # do we check for intermolecular pipi interactions
                input_data[20], # atoms that define benzol ring
                input_data[21], # current date when simulation started
                input_data[22]  # symetryAtoms
            )
            print("Finished - "+str(input_data[5]))
        else:
            with open(os.path.join(pkl), 'rb') as f:
                HBondDescriptor, moleculeData, clusters, looseClusters, internalAngles, relativeOrientation, sphereData, piAverage, piTypes, pipiCounter, rotation, directions, HAverage, HBondsSolvent, Sym, Asym, Density, clusterSphereData = pickle.load(f)
            print("Already done - "+str(input_data[5]))

        ret = {"HBondDescriptor":[input_data[5], HBondDescriptor],
                "moleculeData":[input_data[5], moleculeData],
                "clusters":[input_data[5], clusters],
                "looseClusters":[input_data[5], looseClusters],
                "internalAngles":internalAngles,
                "relativeOrientation":relativeOrientation,
                "sphereData":sphereData,
                "piAverage":piAverage,
                "piTypes":piTypes,
                "pipiCounter":pipiCounter,
                "rotation":rotation,
                "directions":directions,
                "HAverage":HAverage,
                "HBondsSolvent":HBondsSolvent,
                "Sym":Sym, 
                "Asym":Asym,
                "Density":Density,
                "ClusterSphereData":clusterSphereData}
        return ret;

    except Exception:
        print(input_data)
        print("error with: path input now - " + input_data[21] + "  frame - " + str(input_data[5]))
        print(traceback.format_exc())
    gc.collect()

def main():
    now = datetime.today().strftime('%Y-%m-%d')
    pdb_path = "/home/bobrovs/data2/MD2/4ABA/water/ss0.8/4_MD"
    pdb_file = "md.pdb"
    output_path = "/home/bobrovs/data2/MD2/4ABA/water/ss0.8/analysis/Andrievs_analysis"
    mol_atoms = 17 # Number of atoms in the main molecule
    molecules = 18 # Number of the main molecules in each frame
    sol_atoms = 3 # Number of atoms in solvent
    solvent = 32645 # Number of solvent molecules in each frame

    processor_count = 4

    # All atoms except for hydrogens (Used to ignore molecules that are not near this point)
    mol_center_atoms = [0, 1, 2, 3, 5, 7, 9,10, 13, 15]
    sol_center_atoms = [0] #(Used to ignore molecules that are not near this point)

    # highly suggest using a rigid structure like benzol ring
    #First one defined in x-axis direction, second one must not be on the same axis to check orientation
    mol_plane_atoms = [0, 5, 7, 9, 13, 15]

    check_internal = True
    # Two vectors and we messure angle between them each frame for each molecule
    internal_rotation = [[2,3],[5,15]]
    # internal molecule torsion
    # 1st vector that changes direction,
    # 2nd vector give a static direction in the molecule,
    # 3rd vecor is a plane vector in which the torsion is messured.
    internal_torsion = [[3, 4],[0, 1],[1, 3]]

    check_pipi_intermolecular = True
    mol_benzol = [0, 5, 7, 9, 13, 15]

    check_hydrogen_intermolecular = True
    mol_donors = [[3, [4]], [10, [11, 12]]] # Hydrogen donor with their hydrogens molecule
    mol_acceptors = [2, 3, 10] # Hydrogen accpetors molecule

    check_hydrogen_withsolvent = True
    sol_donors = [[0, [1, 2]]] # Hydrogen donor with their hydrogens solvent
    sol_acceptors = [0] # Hydrogen accpetors solvent

    picoseconds = 10 # NUmber of picoseconds in a single frame

    symetryAtoms = ["D3", "A2"]

    # datastore = {"pdb_path": pdb_path,
    #             "pdb_file": pdb_file,
    #             "output_path": output_path,
    #             "molecule_count": molecules,
    #             "molecule_atoms": mol_atoms,
    #             "solvent_count": solvent,
    #             "solvent_atoms": sol_atoms,
    #             "processor_count": processor_count,
    #             "mol_center_atoms": mol_center_atoms,
    #             "sol_center_atoms": sol_center_atoms,
    #             "mol_plane_atoms": mol_plane_atoms,
    #             "check_internal": check_internal,
    #             "internal_rotation": internal_rotation,
    #             "internal_torsion": internal_torsion,
    #             "check_pipi_intermolecular": check_pipi_intermolecular,
    #             "mol_benzol": mol_benzol,
    #             "check_hydrogen_intermolecular": check_hydrogen_intermolecular,
    #             "mol_donors": mol_donors,
    #             "mol_acceptors": mol_acceptors,
    #             "check_hydrogen_withsolvent": check_hydrogen_withsolvent,
    #             "sol_donors": sol_donors,
    #             "sol_acceptors":sol_acceptors,
    #             "picoseconds": picoseconds,
    #             "symetryAtoms": symetryAtoms}

    # with open('input.json', 'w') as f:
    #    json.dump(datastore, f)

    with open('input.json') as f:
        data = json.load(f)
        pdb_path = data["pdb_path"]
        pdb_file = data["pdb_file"]
        output_path = data["output_path"]
        molecules = data["molecule_count"]
        mol_atoms = data["molecule_atoms"]
        solvent = data["solvent_count"]
        sol_atoms = data["solvent_atoms"]
        processor_count = data["processor_count"]
        mol_center_atoms = data["mol_center_atoms"]
        sol_center_atoms = data["sol_center_atoms"]
        mol_plane_atoms = data["mol_plane_atoms"]
        check_internal = data["check_internal"]
        internal_rotation = data["internal_rotation"]
        internal_torsion = data["internal_torsion"]
        check_pipi_intermolecular = data["check_pipi_intermolecular"]
        mol_benzol = data["mol_benzol"]
        check_hydrogen_intermolecular = data["check_hydrogen_intermolecular"]
        mol_donors = data["mol_donors"]
        mol_acceptors = data["mol_acceptors"]
        check_hydrogen_withsolvent = data["check_hydrogen_withsolvent"]
        sol_donors = data["sol_donors"]
        sol_acceptors = data["sol_acceptors"]
        picoseconds = data["picoseconds"]
        symetryAtoms = data["symetryAtoms"]

    splitLen = 5 + 2 + molecules*mol_atoms + solvent*sol_atoms # Number of lines per frame
    outputBase = 'temp' # temp1.txt, temp2.txt, etc.

    #############################################
    #### SPLITING INPUT PDB FILE INTO FRAMES ####
    #############################################

    directory = os.path.join(output_path, "temp-"+now)
    if not os.path.exists(directory):
        os.mkdir(directory)
        frames = 1
        with open(os.path.join(pdb_path, pdb_file), "r") as fin:
            print("opend file - " + str(pdb_file))
            fout = open(os.path.join(output_path, "temp-"+now, outputBase + str(frames) + '.temp'), "w")
            for i, line in enumerate(fin):
                fout.write(line)
                if (i+1)%splitLen == 0:
                    fout.close()
                    frames += 1
                    fout = open(os.path.join(output_path, "temp-"+now, outputBase + str(frames) +
                                             '.temp'), "w")
            fout.close()
            if os.stat(os.path.join(output_path, "temp-"+now, outputBase + str(frames) +
                                    '.temp')).st_size == 0:
                os.remove(os.path.join(output_path, "temp-"+now, outputBase + str(frames) + '.temp'))
    os.chdir(directory)
    files = glob.glob("*.temp")
    frames = len(files)

    ##################################################
    #### READY CONSTANTS THAT THE PROCESSING USES ####
    ##################################################

    out_dir = os.path.join(output_path, "output-"+now)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    input_data = []
    for file in files:
        input_data.append([molecules, mol_atoms, solvent, sol_atoms,
                           os.path.join(output_path, "temp-"+now, file),
                           int(file[4:-5]),
                           output_path,
                           mol_center_atoms, sol_center_atoms, mol_plane_atoms,
                           check_hydrogen_intermolecular, mol_donors, mol_acceptors,
                           check_hydrogen_withsolvent, sol_donors, sol_acceptors,
                           check_internal, internal_rotation, internal_torsion,
                           check_pipi_intermolecular, mol_benzol, now, symetryAtoms
                           ])

    #####################################################
    #### DEFINING LIST TO HOLD ALL FRAME OUTPUT DATA ####
    #####################################################

    moleculeDataSet = []
    internalAngleDataSet = []
    clusterDataSet = []
    looseClusterDataSet = []
    HBondDescriptorSet = []
    relativeOrientations = []
    sphereDataSet = []
    clusterSphereData = []
    piAverages = []
    HAverages = []
    piTypesSet = []
    pipiCounters = {
        "pipiF": 0,
        "pipiT": 0,
        "pipiO": 0,
        "pipiI": 0
        }
    rotations = []
    directionsSet = []
    HBondsSolventSet = []
    SymSet = []
    AsymSet = []
    DensitySet = []

    ##########################################
    #### PARALEL PROCESSING OF EACH FRAME ####
    ##########################################

    print("starting")
    #workerNew(input_data[0])
    if __name__ == '__main__':
        pool_size = processor_count
        pool = Pool(processes=pool_size)
        frame_data_set = pool.map(workerNew, input_data)
        pool.close() # no more tasks
        pool.join()  # wrap up current tasks

    print("finished")

    print("starting joining")
    process = 1
    for frame_data in frame_data_set:
        internalAngleDataSet.extend(frame_data["internalAngles"])
        moleculeDataSet.append(frame_data["moleculeData"])
        clusterDataSet.append(frame_data["clusters"])
        looseClusterDataSet.append(frame_data["looseClusters"])
        HBondDescriptorSet.append(frame_data["HBondDescriptor"])
        relativeOrientations.extend(frame_data["relativeOrientation"])
        sphereDataSet.extend(frame_data["sphereData"])
        clusterSphereData.extend(frame_data["ClusterSphereData"]);
        piAverages.extend([frame_data["piAverage"]])
        HAverages.extend([frame_data["HAverage"]])

        for piType in frame_data["piTypes"]:
            result = next((elem for elem in piTypesSet if elem[0] == piType[0]), None)
            if result == None:
                piTypesSet.append([piType[0], piType[1]])
            else:
                index = piTypesSet.index(result)
                for subelem in piTypesSet[index][1]:
                    value = next((elem[1] for elem in piType[1] if elem[0] == subelem[0]))
                    subelem[1] += value

        pipiCounters["pipiF"] += frame_data["pipiCounter"]["pipiF"]
        pipiCounters["pipiT"] += frame_data["pipiCounter"]["pipiT"]
        pipiCounters["pipiO"] += frame_data["pipiCounter"]["pipiO"]
        pipiCounters["pipiI"] += frame_data["pipiCounter"]["pipiI"]
        rotations.extend(frame_data["rotation"])
        directionsSet.extend(frame_data["directions"])

        for solType in frame_data["HBondsSolvent"]:
            result = next((elem for elem in HBondsSolventSet if elem[0] == solType[0]), None)
            if result == None:
                HBondsSolventSet.append([solType[0], solType[1]])
            else:
                result[1].extend(solType[1])

        SymSet.extend([frame_data["Sym"]])
        AsymSet.extend([frame_data["Asym"]])
        for elem in frame_data["Density"]:
            result = next((x for x in DensitySet if x[0] == elem[0]), None)
            if result == None:
                DensitySet.append([elem[0], elem[2]])
            else:
                index = DensitySet.index(result)
                for x in DensitySet[index][1]:
                    value = next((y[1] for y in elem[2] if y[0] == x[0]))
                    x[1] += value
        print("Finnihes joing frame - " + str(process))
        process += 1

    # # Create PDB files containing central molecule and 1 adjacent molecule
    # filenamesMol = glob.glob("*_mol.pdb")
    # exportPDB(filenamesMol, os.path.join(output_path, "temp-"+now),
    #           os.path.join(output_path, "output-"+now,
    #                        "new_mol_"+now+".pdb"))

    # # Create PDB files containing central molecule and 1 adjacent solvent
    # filenamesSol = glob.glob("*_sol.pdb")
    # exportPDB(filenamesSol, os.path.join(output_path, "temp-"+now),
    #           os.path.join(output_path, "output-"+now,
    #                        "new_sol_"+now+".pdb"))

    out_dir = os.path.join(output_path, "output-"+now)
    os.chdir(out_dir)

    # Write all output text to this file
    DataOutput = os.path.join(output_path, "output-"+now, "new_data_"+now+".txt")

    ########################################
    #### MOLECULE INTERNAL INTERACTIONS ####
    ########################################

    # Calculate carboxil groups rotational angle probability
    fig, ax = plt.subplots()
    ang = [r["internalAngle"] for r in internalAngleDataSet]
    bin_centers, kde = vonmises_fft_kde(ang, 20, 100, 0, 2*np.pi)
    kde_max = max(kde)
    ax.plot(bin_centers, kde)
    #sns.distplot(ang, ax=ax, kde=True, hist=False, norm_hist=True, kde_kws = {'linewidth': 3});
    ax.set_ylabel('Probability, %')
    ax.set_xlabel('Angle, rad')
    ax.set_title('Carboxil groups rotational position probability density')
    ax.grid(axis='y', alpha=0.75)
    ax.set_facecolor('#d8dcd6')
    ax.set_xlim(left=0, right=2*np.pi)
    ticks_y = tkr.FuncFormatter(lambda y, pos: '{0:g}'.format(to_energy(y, kde_max)))
    ax.yaxis.set_major_formatter(ticks_y)
    fig.savefig('Carboxil group rotation angle.pdf', format='pdf')

    temp = [r for r in internalAngleDataSet if r["internalAngle"] <= np.pi]
    rotation_durration = np.empty([molecules, frames])
    for val in temp:
        rotation_durration[val["molecule"]][val["frame"]-1] = val["internalAngle"]

    average_rotation_time_s = 0.0
    average_rotation_time_l = 0.0
    s = 0
    l = 0
    for molec in rotation_durration:
        first = True
        small_angle = False
        count = 0
        for frame_data in molec:
            if first:
                first = False
                if abs(frame_data) > np.pi/2:
                    small_angle = False
                else:
                    small_angle = True
                count += 1
            else:
                if abs(frame_data) > np.pi/2:
                    if not small_angle:
                        count += 1
                    else:
                        average_rotation_time_s = (average_rotation_time_s * s + count) / (s+1)
                        s += 1
                        small_angle = False
                        count = 1
                else:
                    if small_angle:
                        count += 1
                    else:
                        average_rotation_time_l = (average_rotation_time_l * l + count) / (l+1)
                        l += 1
                        small_angle = True
                        count = 1
    saveText("Average duration for when the internal angle is smaller than pi/2 is - " +
             str(average_rotation_time_s*picoseconds) + " +/- " + str(picoseconds) +
             " picoseconds", DataOutput)
    saveText("Average duration for when the internal angle is larger than pi/2 is - " +
             str(average_rotation_time_l*picoseconds) + " +/- " + str(picoseconds) +
             " picoseconds", DataOutput)

    #Calculate hygrogen bond direction dependant on carboxil group rotation
    int_angle = [r["internalAngle"] for r in internalAngleDataSet]
    int_torsion = [r["internalTorsion"] for r in internalAngleDataSet]
    Spherical_KDE('Hydrogen direction depending on carboxil groups rotation.pdf',
                  'Hydrogen direction depending on carboxil groups rotation',
                  "Carboxil groups rotational angle, rad", "Carboxil groups hygrogens torsion, rad",
                  int_angle, 0, 2*np.pi, int_torsion, -np.pi, np.pi, "viridis", 1.4)

    ###########################################
    #### MOLECULE TO MOLECULE INTERACTIONS ####
    ###########################################

    HBondSetDistribution = []
    existingHBondSets = []
    currentHBondSets = []
    existingPiPiTypes = []
    currentPiPiTypes = []
    existingPiPi = []
    currentPiPi = []
    existingHBonds = []
    currentHBonds = []
    i = 0

    moleculeDataSet.sort()
    for frame in moleculeDataSet:
        print("Processing frame - " + str(frame[0]));
        for mol in frame[1]:
            # Checking PiPi interaction between molecules
            if mol["isPiPiTight"]:
                elem = next((x for x in currentPiPi if x[0] == mol["pairName"]), None)
                if elem == None:
                    currentPiPi.append([mol["pairName"], 1, i])
                else:
                    elem[1] += 1
                    elem[2] = i

                elem = next((x for x in currentPiPiTypes if x[0] == mol["pipiType"]), None)
                if elem == None:
                    currentPiPiTypes.append([mol["pipiType"], [[mol["pairName"], 1, i]]])
                else:
                    sub = next((x for x in elem[1] if x[0] == mol["pairName"]), None)
                    if sub == None:
                        elem[1].append([mol["pairName"], 1, i])
                    else:
                        sub[1] += 1
                        sub[2] += i
            elif mol["isPiPiLoose"]:
                elem = next((x for x in currentPiPi if x[0] == mol["pairName"]), None)
                if elem != None:
                    elem[1] += 1
                    elem[2] = i
                elem = next((x for x in currentPiPiTypes if x[0] == mol["pipiType"]), None)
                if elem != None:
                    sub = next((x for x in elem[1] if x[2] and x[0] == mol["pairName"]), None)
                    if sub != None:
                        sub[1] += 1
                        sub[2] += i

            # Checking Hydrogen bond interaction between molecules
            if mol["isHBondTight"]:
                elem = next((x for x in currentHBonds if x[0] == mol["pairName"]), None)
                if elem == None:
                    currentHBonds.append([mol["pairName"], 1, i])
                else:
                    elem[1] += 1
                    elem[2] = i
                for HBondSet in mol["HBondSets"]:
                    elem = next((x for x in HBondSetDistribution if (x[0] == HBondSet[0] and
                                                                     x[1] == HBondSet[1] or
                                                                     x[1] == HBondSet[0] and
                                                                     x[0] == HBondSet[1])),
                                None)
                    if elem == None:
                        HBondSetDistribution.append([HBondSet[0], HBondSet[1], 1])
                    else:
                        elem[2] += 1

                    elem = next((x for x in currentHBondSets if x[0] == mol["pairName"] and
                                 (x[1] == HBondSet[0] and x[2] == HBondSet[1] or x[2] == HBondSet[0] and
                                  x[1] == HBondSet[1])), None)
                    if elem == None:
                        currentHBondSets.append([mol["pairName"], HBondSet[0],
                                                  HBondSet[1], 1, i])
                    else:
                        elem[3] += 1
                        elem[4] += i
            elif mol["isHBondLoose"]:
                elem = next((x for x in currentHBonds if x[0] == mol["pairName"]), None)
                if elem != None:
                    elem[1] += 1
                    elem[2] = i
                for HBondSet in mol["HBondSets"]:
                    elem = next((x for x in currentHBondSets if x[0] == mol["pairName"] and
                                 (x[1] == HBondSet[0] and x[2] == HBondSet[1] or x[2] == HBondSet[0] and
                                  x[1] == HBondSet[1])), None)
                    if elem != None:
                        elem[3] += 1
                        elem[4] += i

        for types in currentPiPiTypes:
            for temp in types[1]:
                if temp[2] != i:
                    elem = next((x for x in existingPiPiTypes if x[0] == types[0]), None)
                    if elem == None:
                        existingPiPiTypes.append([types[0], [temp]])
                    else:       
                        elem[1].append(temp)
                    types[1].remove(temp)
        for temp in currentPiPi:
            if temp[2] != i:
                existingPiPi.append(temp)
                currentPiPi.remove(temp)
        for temp in currentHBonds:
            if temp[2] != i:
                existingHBonds.append(temp)
                currentHBonds.remove(temp)
        for temp in currentHBondSets:
            if temp[4] != i:
                existingHBondSets.append(temp)
                currentHBondSets.remove(temp)
        i += 1
    
    for types in currentPiPiTypes:
        for temp in types[1]:
            elem = next((x for x in existingPiPiTypes if x[0] == types[0]), None)
            if elem == None:
                existingPiPiTypes.append([types[0], [temp]])
            else:       
                elem[1].append(temp)
            types[1].remove(temp)
    for temp in currentPiPi:
        existingPiPi.append(temp)
        currentPiPi.remove(temp)
    for temp in currentHBonds:
        existingHBonds.append(temp)
        currentHBonds.remove(temp)
    for temp in currentHBondSets:
        existingHBondSets.append(temp)
        currentHBondSets.remove(temp)
    
    print("Processing done");
    
    ##############################################################
    # Drawing distance rependancy on relative angle of molecules #
    ##############################################################

    planeAngles = [r["orientation"] for r in relativeOrientations]
    distances = [r["distance"] for r in relativeOrientations]
    Cylindrical_KDE('Distance_per_orientation.pdf',
                    'Distance_per_orientation',
                    'Probablity density of angles, rad',
                    'Probability density of distance, A',
                    planeAngles, 0, np.pi, distances, 3, 11, "viridis", 1.2)

    ###################################################################################################
    # Drawing realtion between polar angle and polar torsion of adjacent molecules position using KDE #
    ###################################################################################################

    ang_list = [r["CentralPolarAngle"] for r in sphereDataSet]
    tor_list = [r["CentralPolarTorsion"] for r in sphereDataSet]
    colors = [r["AdjacentPolar"] for r in sphereDataSet]
    Spherical_KDE('Flat_sphere_KDE.pdf',
                  'Flat sphere KDE',
                  'Angle between projection of line between molecules in XZ-plane and the X-Axis, rad',
                  'Angle between Y-Axis and line between molecules, Rad',
                  tor_list, -np.pi, np.pi, ang_list, 0, np.pi, "viridis", 1.4)

    #########################################################################################
    # Drawing second molecules postion in polar coordinates color represnts its orientation #
    #########################################################################################

    #Import data
    ang_list = [r["CentralPolarAngle"] for r in clusterSphereData]
    tor_list = [r["CentralPolarTorsion"] for r in clusterSphereData]
    colors = [r["AdjacentPolar"] for r in clusterSphereData]
    
    Spherical_KDE('Flat_sphere_KDE_cluster.pdf',
                  'Flat sphere KDE cluster',
                  'Angle between projection of line between molecules in XZ-plane and the X-Axis, rad',
                  'Angle between Y-Axis and line between molecules, Rad',
                  tor_list, -np.pi, np.pi, ang_list, 0, np.pi, "viridis", 1.4)

    
    theta = tor_list
    phi = ang_list
    xx = np.sin(phi)*np.cos(theta)*1.00
    yy = np.sin(phi)*np.sin(theta)*1.00
    zz = np.cos(phi)*1.00

    df = pd.DataFrame({'X': xx, 'Y': yy, 'Z': zz})

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'], c=colors, s=1)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    i = 0
    ax.view_init(30, i)
    fig.savefig('Cluster points on sphere.pdf', format='pdf')
    plt.show()

    #########################################################################################
    ######################## Drawing Pi-Pi type depending on distance #######################
    #########################################################################################

    # Create plot
    fig, ax = plt.subplots()
    for temp in piTypesSet:
        x = [r[0] for r in temp[1]]
        y = [r[1] for r in temp[1]]
        df = pd.DataFrame({"x":x, "y":y})
        group = temp[0]
        ax.plot(x, y, label=group)
    ax.set_xlim(left=0)
    plt.title("Pi-Pi stacking type depending on distance")
    plt.ylabel("Pi-Pi type probability density")
    plt.xlabel("Distance, A")
    plt.legend(loc=1)
    fig.savefig('Pi-Pi type depending on distance.pdf', format='pdf')
    plt.show()

    ##############################################################
    # Calculate probobilty of molecule having a PiPI interaction #
    ##############################################################

    total = 0
    for elem in piAverages:
        total += elem

    total = total/frames
    saveText("Average Pi-Pi bonds per molecule per frame - " +
             str(total) + " +/- " + str(1/frames), DataOutput)

    ##############################################################
    # Calculate probobilty of molecule having a Hydrogen bond    #
    ##############################################################

    total = 0
    for elem in HAverages:
        total += elem

    total = total/frames
    saveText("Average inermolecular Hydrogen bonds per molecule per frame - " +
             str(total) + " +/- " + str(1/frames), DataOutput)

    ####################################################################################################
    # Calculate probobilty of molecule rotation in cases of face to face and offfset pi-pi intercations#
    ####################################################################################################

    fig, ax = plt.subplots()
    bin_centers, kde = vonmises_fft_kde(rotations, 10, 100, -np.pi, np.pi)
    kde_max = max(kde)
    print(kde_max)
    ax.plot(bin_centers, kde)

    ax.set_ylabel('Probability, %')
    ax.set_xlabel('Angle, rad')
    ax.set_title('Density of molecule rotation of Face to Face and Offset pi-pi interactions')
    ax.grid(axis='y', alpha=0.75)
    ax.set_facecolor('#d8dcd6')
    ax.set_xlim(left=-np.pi, right=np.pi)
    ax.set_ylim(bottom=0)
    ticks_y = tkr.FuncFormatter(lambda y, pos: '{0:g}'.format(to_energy(y, kde_max)))
    ax.yaxis.set_major_formatter(ticks_y)
    fig.savefig('Density of molecule relative rotation.pdf', format='pdf')

    ####################################################################################
    # Calculate probobilty of molecule direction in cases of TShape pi-pi intercations #
    ####################################################################################
    fig, ax = plt.subplots()
    bin_centers, kde = vonmises_fft_kde(directionsSet, 10, 40, 0, np.pi)
    kde_max = max(kde)
    ax.plot(bin_centers, kde)

    ax.set_ylabel('Probability, %')
    ax.set_xlabel('Angle, rad')
    ax.set_title('Probability density of molecule direction of TShape pi-pi interactions')
    ax.grid(axis='y', alpha=0.75)
    ax.set_facecolor('#d8dcd6')
    ax.set_xlim(left=0, right=np.pi)
    ticks_y = tkr.FuncFormatter(lambda y, pos: '{0:g}'.format(to_energy(y, kde_max)))
    ax.yaxis.set_major_formatter(ticks_y)
    fig.savefig('TShape dirrection.pdf', format='pdf')

    #######################################################
    # Calculate probobilty of diffrent pi-pi interactions #
    #######################################################

    pipiTotal = 0
    for value in pipiCounters.values():
        pipiTotal += value
    saveText('PiPi interactions: TShape - ' + "%.3f" % (pipiCounters["pipiT"]/pipiTotal*100) +
             '%'  + " +/- " + "%.3f" % (1/pipiTotal*100) + '%  Face to Face - ' +
             "%.3f" % (pipiCounters["pipiF"]/pipiTotal*100) + '%'  + " +/- " +
             "%.3f" % (1/pipiTotal*100) + '% Offset - ' +
             "%.3f" % (pipiCounters["pipiO"]/pipiTotal*100) +
             '%'  + " +/- " + "%.3f" % (1/pipiTotal*100) + '% Intermidiary - ' +
             "%.3f" % (pipiCounters["pipiI"]/pipiTotal*100) + '%'  + " +/- " +
             "%.3f" % (1/pipiTotal*100) + '%', DataOutput)

    ###########################################################
    ### Calculate average durration of pipi interactions ######
    ###########################################################

    pipiDurationAvr = 0
    i = 0
    for bond in existingPiPi:
        dur = bond[1]+1
        pipiDurationAvr = (pipiDurationAvr*i + dur)/(i+1)
        i += 1
    saveText("PiPi bond average duration - " +
             str(pipiDurationAvr * picoseconds) + " +/- " +
             str(picoseconds) + " picoseconds", DataOutput)

    ################ FIX THIS LATER ##############################

    # Pi-Pi bond durration density
    fig, ax = plt.subplots()
    pipibondDur = [r[1] * picoseconds for r in existingPiPi]  
    sns.distplot(pipibondDur, ax=ax, kde=False, hist=True,
                            norm_hist=True, kde_kws={'linewidth': 3})
    ax.set_ylabel('Probablity density')
    ax.set_xlabel('Durration, ps')
    ax.set_title('Pi-Pi interaction duration distribution')
    ax.grid(axis='y', alpha=0.75)
    ax.set_facecolor('#d8dcd6')  
    fig.savefig('Pi-Pi bond durration density.pdf', format='pdf')

    fig, ax = plt.subplots()
    for pipiType in existingPiPiTypes:
        pipiDurationAvr = 0
        i = 0
        for bond in pipiType[1]:
            dur = bond[1]+1
            pipiDurationAvr = (pipiDurationAvr*i + dur)/(i+1)
            i += 1
        saveText("PiPi bond type "+str(pipiType[0])+" average duration - " +
                 str(pipiDurationAvr * picoseconds) + " picoseconds", DataOutput)
        pipibondDur = [r[1] * picoseconds for r in pipiType[1]]
        sns.distplot(pipibondDur, ax=ax, kde=True, hist=True, norm_hist=True,
                     label=str(pipiType[0]), kde_kws={'linewidth': 3})
    ax.set_ylabel('Probability density')
    ax.set_xlabel('Durration, ps')
    ax.set_title('Pi-Pi interaction duration distribution')
    ax.grid(axis='y', alpha=0.75)
    ax.set_facecolor('#d8dcd6')
    fig.savefig('Pi-Pi bond durration density per type.pdf', format='pdf')

    pipiDurationMax = [0, 0]
    i = 0
    for bond in existingPiPi:
        dur = bond[1]+1
        stamp = bond[2]
        if pipiDurationMax[0] < dur:
            pipiDurationMax[0] = dur
            pipiDurationMax[1] = stamp
    saveText("Longest lasting PiPi interaction lasted " +
             str(pipiDurationMax[0] * picoseconds) + " +/- " +
             str(picoseconds) + " picoseconds, last seen on frame - " +
             str(pipiDurationMax[1]+1), DataOutput)

    ###########################################################
    ### Calculate average durration of hydrogen bonding #######
    ###########################################################

    HBondDurationAvr = 0
    i = 0
    for bond in existingHBonds:
        dur = bond[1]+1
        HBondDurationAvr = (HBondDurationAvr*i + dur)/(i+1)
        i += 1
    saveText("Hydrogen bond average duration - " +
             str(HBondDurationAvr * picoseconds) + " +/- " +
             str(picoseconds) + " picoseconds", DataOutput)

    HBondDurationMax = [0, 0]
    i = 0
    for bond in existingHBonds:
        dur = bond[1]+1
        stamp = bond[2]
        if HBondDurationMax[0] < dur:
            HBondDurationMax[0] = dur
            HBondDurationMax[1] = stamp
    saveText("Longest lasting Hydrogen interaction lasted " +
             str(HBondDurationMax[0] * picoseconds) + " +/- " +
             str(picoseconds) + " picoseconds, last seen on frame - " +
             str(HBondDurationMax[1]+1), DataOutput)

    TotalCount = 0
    for values in HBondSetDistribution:
        TotalCount += values[2]

    for values in HBondSetDistribution:
        saveText("Hydrogen bond between " + str(values[0]) +
                 " and " + str(values[1]) + " are " +
                 str(values[2]/TotalCount) + " +/- " +
                 str(1/TotalCount) + " of hydrogen bonds", DataOutput)

    bondTypeDurationAvr = []
    for bond in existingHBondSets:
        hb_t1 = bond[1]
        hb_t2 = bond[2]
        duration = bond[3]
        elem = next((x for x in bondTypeDurationAvr if
                     (x[0] == hb_t1 and x[1] == hb_t2 or x[0] == hb_t2 and x[1] == hb_t1)),
                    None)
        if elem == None:
            bondTypeDurationAvr.append([hb_t1, hb_t2, duration, 1])
        else:
            elem[2] = (elem[2] * elem[3] + duration)/elem[3]
            elem[3] += 1

    for hb_type in bondTypeDurationAvr:
        saveText("Average hydrogen bond duration between " +
                 str(hb_type[0]) + " and " + str(hb_type[1]) +
                 " was " + str(hb_type[2] * picoseconds) + " +/- " +
                 str(picoseconds) + " picoseconds", DataOutput)

    ###########################################
    #### MOLECULE CLUSTERING CALCULATIONS  ####
    ###########################################

    clusterDataSet.sort()
    looseClusterDataSet.sort()
    existingClusters = []
    currentClusters = []
    i = 0
    for frame in clusterDataSet:
        for cluster in frame[1]:
            elem = next((x for x in currentClusters if x[0] == cluster), None)
            if elem == None:
                size = len(cluster)
                currentClusters.append([cluster, size, 0, i])
            else:
                extended = False
                for clusterLoose in looseClusterDataSet[i][1]:
                    found = True
                    for molecule in cluster:
                        if not molecule in clusterLoose:
                            found = False
                            break
                    if found:
                        extended = True
                        elem[2] += 1
                        elem[3] = i
                if not extended:
                    existingClusters.append(temp)
                    currentClusters.remove(temp)
        for temp in currentClusters:
            if temp[3] != i:
                existingClusters.append(temp)
                currentClusters.remove(temp)
        i += 1

    ###########################################################
    ### Calculate average durration of each n-mer        ######
    ###########################################################

    sizeDurationAvr = []
    i = 0
    for cluster in existingClusters:
        size = cluster[1]
        dur = cluster[2]+1
        elem = next((x for x in sizeDurationAvr if x[0] == size), None)
        if elem == None:
            sizeDurationAvr.append([size, dur])
        else:
            elem[1] = (elem[1]*i + dur)/(i+1)
        i += 1
    for sizeType in sizeDurationAvr:
        saveText(str(sizeType[0])+"-mer average - " +
                 str(sizeType[1] * picoseconds) + " +/- " +
                 str(picoseconds) + " picoseconds", DataOutput)

    ###########################################################
    ### Calculate maximum durration of each n-mer        ######
    ###########################################################

    sizeDurationMax = []
    for cluster in existingClusters:
        size = cluster[1]
        dur = cluster[2]+1
        stamp = cluster[3]
        elem = next((x for x in sizeDurationMax if x[0] == size), None)
        if elem == None:
            sizeDurationMax.append([size, dur, stamp])
        elif elem[1] < dur:
            elem[1] = dur
            elem[2] = stamp

    for sizeType in sizeDurationMax:
        saveText("Longest lasting " + str(sizeType[0])+"-mer lasted " +
                 str(sizeType[1] * picoseconds) + " +/- " + str(picoseconds) +
                 " picoseconds, last seen on frame - " +
                 str(sizeType[2]+1), DataOutput)

    #############################################################
    ### Calculating precentige of molecules in n-mer clusters ###
    #############################################################

    clusterSize = np.zeros(molecules+1)
    i = 0
    for frame in clusterDataSet:
        tempList = np.zeros(molecules+1)
        count = 0
        for cluster in frame[1]:
            size = len(cluster)
            tempList[size] += (size/molecules)
            count += size
        tempList[1] = ((molecules-count)/molecules)
        j = 0
        while j < molecules+1:
            val = clusterSize[j]
            val = (val*i + tempList[j])/(i+1)
            clusterSize[j] = val
            j += 1
        i += 1

    x = []
    y = []
    i = 0
    while i < molecules+1:
        val = clusterSize[i]
        if val != 0:
            x.append(i)
            y.append(val)
        i += 1

    fig, ax = plt.subplots()
    ax.bar(x, y)
    y_max = max(y)
    ticks_y = tkr.FuncFormatter(lambda y, pos: '{0:g}'.format(to_energy(y, y_max)))
    ax.yaxis.set_major_formatter(ticks_y)
    plt.title("Molecule distribution in clusters")
    plt.ylabel("Probablity of belonging in cluster")
    plt.xlabel("Cluster size")
    fig.savefig('Molecule distribution in clusters.pdf', format='pdf')

    #################################################
    ### INTERACTION BETWEEN SOLVENT AND MOLECULES ###
    #################################################

    HBondsSolventTime = []
    i = 0
    HCount = 0
    HAvr = 0

    for frame in HBondDescriptorSet:
        HBondsSolventTime.append([frame[0], []])
        temp = []
        HCount = 0
        for descriptor in frame[1]:
            name = descriptor[0]
            HCount += descriptor[1]
            if name not in temp:
                temp.append(name)
                HBondsSolventTime[i][1].append([name, descriptor[1]])
            else:
                index = temp.index(name)
                HBondsSolventTime[i][1][index][1] += descriptor[1]
        HBondsSolventTime[i][1].sort()
        HAvr = (HAvr * i + HCount/molecules)/(i+1)
        i += 1
    saveText("Hydrogen bond with count solvent per molecule per frame - "+str(HAvr), DataOutput)

    TypeHBondsSolventTime = []
    for frame in HBondsSolventTime:
        time = frame[0]
        bond_types = frame[1]
        for bond_type in bond_types:
            result = next((elem for elem in TypeHBondsSolventTime if bond_type[0] == elem[0]), None)
            if result == None:
                TypeHBondsSolventTime.append([bond_type[0], [[time, bond_type[1]]]])
            else:
                index = TypeHBondsSolventTime.index(result)
                TypeHBondsSolventTime[index][1].append([time, bond_type[1]])

    # Create plot
    fig, ax = plt.subplots()
    i = 0
    for bond_type in TypeHBondsSolventTime:
        x = [r[0]*picoseconds/1000 for r in bond_type[1]]
        y = [r[1]/molecules for r in bond_type[1]]
        df = pd.DataFrame({"x":x, "y":y})
        group = bond_type[0]
        sns.regplot("x", "y", data=df, scatter=False, color=".1")
        slope, intercept, r_value, p_value, std_err = ss.linregress(x=x, y=y)
        saveText("y = "+str(round(slope, 2))+"*x + "+str(round(intercept, 2))+
                 " R^2="+str(round(r_value**2, 2)) + " Std = " + str(round(std_err, 2)), DataOutput)
        i += 1
    ax.set_xlim(left=0)
    plt.title("Hydrogen bonded solvent molecule count")
    plt.ylabel("Solvent molecule count")
    plt.xlabel("Nanoseconds")
    plt.legend(loc=1)
    fig.savefig('Hydrogen bonded solvent molecule count.pdf', format='pdf')

    totalS = 0
    for elem in SymSet:
        totalS += elem

    totalA = 0
    for elem in AsymSet:
        totalA += elem

    ratioS = totalS/(totalS+totalA)
    ratioA = totalA/(totalS+totalA)
    saveText("Of hydrogen bonded molecule pairs " + str(ratioS) +
             " are sysmtricaly hydrogen bonded and " + str(ratioA) +
             " are asymetrically bonded.", DataOutput)

    # Create plot
    fig, ax = plt.subplots()
    for temp in DensitySet:
        x = [r[0] for r in temp[1]]
        y = [r[1] for r in temp[1]]
        df = pd.DataFrame({"x":x, "y":y})
        group = temp[0]
        ax.plot(x, y, label=group)
    ax.set_xlim(left=0)
    plt.title("Solvent probabilty density from each acceptor/donor")
    plt.ylabel("Solvent density")
    plt.xlabel("Distance, A")
    plt.legend(loc=1)
    fig.savefig('Solvent probabilty density from each acceptor and donor.pdf', format='pdf')

    fig, ax = plt.subplots()
    for types in HBondsSolventSet:
        HAngles = []
        distances = []
        for data in types[1]:
            name = types[0]
            distances.append(data[0])
            HAngles.append(data[1])

        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))

        Flat_KDE('Solvent density from '+name+'.pdf',
              'Solvent density from '+name,
              'Probablity density of angles, rad',
              'Probability density of distance, A',
              HAngles, 2.3, np.pi, distances, 2.5, 3.2, "viridis", 1.4);
#        df = pd.DataFrame({"Probablity density of angles, rad":HAngles,
#                           "Probability density of distance, A":distances})
#
#        sns.set(style="white", color_codes=True)
#        ax = sns.jointplot(data=df, x='Probablity density of angles, rad',
#                           y='Probability density of distance, A', kind='kde',
#                           cbar=True, cbar_kws={"format": formatter}, color="skyblue") #n_levels=20,
#        ax.ax_joint.axvline(x=np.pi, linestyle='--')
#        ax.ax_joint.axvline(x=2.268, linestyle='--')
#        ax.ax_joint.axhline(y=3.2, linestyle='--')
#        fig = ax.get_figure()
#        fig.savefig('Solvent density from '+name+'.pdf', format='pdf')

main()
print("Done")
