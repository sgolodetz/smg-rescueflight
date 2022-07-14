#!/usr/bin/python
#
# Requirements: 
# sudo apt-get install python-argparse


# Copyright (c) 2013 Thomas Whelan
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#

import sys
import numpy
import argparse

if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script summarises the vertex-wise error computed from a CloudCompare save file. 
    ''')
    parser.add_argument('file', help='vertices saved in ASCII format from CloudCompare')
    args = parser.parse_args()

    dataColumn = -1
    numVertices = 0
    dataArray = numpy.zeros((0, 0))
    
    with open(args.file, 'r') as f:
        first_line = f.readline()
        
        tokens = first_line.split()
        
        if tokens[0] != "//X":
            print("Parse error: First line should be column labels")
            sys.exit(-1)
        else:
            for index, item in enumerate(tokens):
                if item == "C2C_absolute_distances":
                    dataColumn = index
                    
        second_line = f.readline()
        
        tokens = second_line.split()
        
        if len(tokens) != 1:
            print("Parse error: Second line should be number of vertices")
            sys.exit(-1)
        else:
            numVertices = int(tokens[0])
        
        dataArray = numpy.zeros((numVertices, 1))

        currentIndex = 0

        for line in f:
            tokens = line.split()
            
            dataArray[currentIndex] = float(tokens[dataColumn])
            
            currentIndex = currentIndex + 1

    print("vertices %d" % numVertices)
    print("mean %f m" % numpy.mean(dataArray))
    print("median %f m" % numpy.median(dataArray))
    print("std %f m" % numpy.std(dataArray))
    print("min %f m" % numpy.min(dataArray))
    print("max %f m" % numpy.max(dataArray))
