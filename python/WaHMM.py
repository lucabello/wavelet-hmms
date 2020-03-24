#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys

# Wrapper around compiled C++ WaHMM

arguments = sys.argv
arguments[0] = "bin/WaHMM"

#print("###START###")
subprocess.call(arguments)
#print("###END###")
