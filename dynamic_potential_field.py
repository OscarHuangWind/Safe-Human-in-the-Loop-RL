#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:16:52 2023

@author: oscar
"""

import math
import numpy as np

from obstacle import Obstacle

class FDPF():
    def __init__(self):
        self.Upsilon_coeff_s = 0.3 # m/s
        self.Upsilon_coeff_l = 0.7 # m/s
        self.major_scale = 2.0
        self.semi_scale = 0.15
        self.decay_coeff = 0.5
        
        self.vehicle_width = None
        self.lane_width = None
        self.margin_distance = 1.0
        
    def update(self, input):
        self.neighborhood_grids = input

    def getIntensityAt(self, ego_s, ego_l, ego_delta_yaw, ego_v):
        intensity_s = 0
        intensity_l = 0
        eog_v_s = ego_v * math.cos(ego_delta_yaw)
        eog_v_l = ego_v * math.sin(ego_delta_yaw)
        
        for (s_index, l_index) in [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]:
            
            neighborhood_grid = self.neighborhood_grids[s_index][l_index]
            
            for (s_i, l_i, delta_yaw_i, v_i) in neighborhood_grid:
                relative_direction = math.atan2(l_i - ego_l, s_i - ego_s) #从交通车指向本车
                obstacle_v_s = v_i * math.cos(delta_yaw_i)
                obstacle_v_l = v_i * math.sin(delta_yaw_i)
                relative_velocity_s = max(eog_v_s - obstacle_v_s, 0)
                relative_velocity_l = max(eog_v_l - obstacle_v_l, 0)
                coefficient_s = self.Upsilon_coeff_s / (self.major_scale * self.Upsilon_coeff_s + relative_velocity_s)
                coefficient_l = self.Upsilon_coeff_l / (self.semi_scale * self.Upsilon_coeff_l + relative_velocity_l)
                
                EUCLEADIAN_distance = math.sqrt((ego_s - s_i)**2 * coefficient_s + (ego_l - l_i)**2 * coefficient_l)
                intensity = np.exp(-EUCLEADIAN_distance * self.decay_coeff) 
                
                intensity_s += intensity * math.cos(relative_direction)
                intensity_l += intensity * math.sin(relative_direction)
        
        self.left_boundry = self.lane_width * 1.5 - self.margin_distance
        self.right_boundry = - self.lane_width * 1.5 + self.margin_distance

        left_off_road_distance =  max(0, ego_l - self.left_boundry)
        right_off_road_distance =  min(0, ego_l - self.right_boundry)
        
        intensity_left_bound = left_off_road_distance**2
        intensity_right_bound = right_off_road_distance**2
        intensity_bound = intensity_left_bound - intensity_right_bound
        intensity_l_with_bound = intensity_l + intensity_left_bound - intensity_right_bound
                
        frenet_direction = math.atan2(intensity_l, intensity_s)
        intensity = math.sqrt(intensity_s**2 + intensity_l**2)
        
        frenet_direction = math.atan2( intensity_l_with_bound, intensity_s)
        intensity_with_bound = math.sqrt(intensity_s**2 + intensity_l_with_bound**2)
        
        return (intensity, intensity_with_bound, frenet_direction,
                intensity_s, intensity_l, intensity_bound)