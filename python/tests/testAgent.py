#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:58:00 2023

@author: dbellis
"""
def write_output(filename, output):
    output_file = open(filename, 'w')
    output_file.write(output)
    output_file.close()

def get_with_precision(n):
    return f"{n:.16f}"

def get_output_2d(mat):
    d0, d1 = mat.shape
    ss = ""
    ss += "[\n"
    for x in mat:
        ss += "  ["
        iy = 0
        for y in x:
            ss += get_with_precision(y)
            if not iy == (d1 - 1):
                ss += ","
            iy += 1
        ss += "]\n"
    ss += "]\n"
    return ss

def get_output_3d(aT):
    d0, d1, d2 = aT.shape
    ss = ""
    ss += "(" + str(d0) + "," + str(d1) + "," + str(d2) + ")=\n[\n"
    for x in aT:
        ss += "  [\n"
        for y in x:
            ss += "    ["
            iz = 0
            for z in y:
                ss += get_with_precision(z)
                if not iz == (d2 - 1):
                    ss += ","
                iz += 1
            ss += "]\n"
        ss += "  ]\n"
    ss += "]\n"
    return ss

def get_output_4d(aT):
    d0 = len(aT)
    if d0 == 0:
        return ""
    d1, d2, d3 = aT[0].shape
    ss = ""
    ss += "(" + str(d0) + "," + str(d1) + "," + str(d2) + "," + str(d3) + ")=\n[\n"
    for w in aT:
        ss += "  [\n"
        for x in w:
            ss += "    [\n"
            for y in x:
                ss += "      ["
                iz = 0
                for z in y:
                    ss += get_with_precision(z)
                    if not iz == (d3 - 1):
                        ss += ","
                    iz += 1
                ss += "]\n"
            ss += "    ]\n"
        ss += "  ]\n"
    ss += "]\n"
    return ss
