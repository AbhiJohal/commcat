# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:56:47 2020

@author: Abhi
"""

def seq(filename):
    
    f= open(filename,"w+")
    
    for line in f:
        l = line.split()
        m = rangeCount(l[1], l[2])
        line = line.split() + m
        f.write(line)
    
    print(f.readlines())
    f.close
        

def count(n):
    
    c = 0
    
    while n > 1:
        if n % 2 == 0:
            n = n/2 
            c = c + 1
        else:
            n = 3*n + 1
            c = c+1
    return c
    
def rangeCount(start, end):
    
    t = {1:1} #dictionary
    
    m = 0 #max
    
    for i in range(start, end+1):
        
        t[i] = count(i)
        m = max(m, t[i])
        
    return m