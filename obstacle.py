#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:31:21 2023

@author: oscar
"""

class Obstacle(object):
    def __init__(self):
        self.s = None
        self.l = None
        self.yaw = None
        self.v = None
    
    def setS(self, s):
        self.s = s
    
    def getS(self):
        return self.relative_s

    def setL(self, l):
        self.l = l

    def getL(self):
        return self.l
    
    def setYaw(self, yaw):
        self.yaw = yaw
        
    def getYaw(self):
        return self.yaw
    
    def setV(self, v):
        self.v = v
        
    def getV(self):
        return self.v
    
    def setProperty(self, s, l, yaw, v):
        self.s = s
        self.l = l
        self.yaw = yaw
        self.v = v
        
    def getProperty(self):
        return self.s, self.l, self.yaw, self.v