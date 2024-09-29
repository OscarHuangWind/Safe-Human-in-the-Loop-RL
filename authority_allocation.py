#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:18:54 2023

@author: oscar
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from scipy.special import softmax
from termcolor import colored

from dynamic_potential_field import *
from common import *

class Arbitrator(object):
    def __init__(self):
        self.shared_control = False
        self.rl_authority = 1.0
        self.human_authority = 0.0
        self.coeff = 10.0
        
        self.ego_grid_s_lenght = 10
        self.perception_grid_s_lenght = 30
        self.lane_num = 3
        self.lane_center = [-3.2, 0, 3.2]
        
        self.ego_vehicle = Vehicle("./ego_vehicle.yaml")
        self.field = FDPF()
        self.field.vehicle_width = self.ego_vehicle.param.W
        self.curvature_smoother = CurvatureSmoother(50)
        
    def authority(self, obs, rl_a, human_a):
        if not self.shared_control:
            return 0.0 # full authority to human
        
        lane_index= 1
        self.reference_line = Trajectory()
        self.num_trajectory_points = len(obs.waypoint_paths[lane_index])
    
        for i in range(self.num_trajectory_points):
            if i == 0:
                yaw=SMARTS_yawCorrect(obs.waypoint_paths[lane_index][i].heading, obs.waypoint_paths[lane_index][i].heading)
            else:
                yaw=SMARTS_yawCorrect(obs.waypoint_paths[lane_index][i].heading, self.reference_line.points[-1].yaw)
            p = TrajPoint(x=obs.waypoint_paths[lane_index][i].pos[0], y=obs.waypoint_paths[lane_index][i].pos[1], yaw = yaw)
            if i < self.num_trajectory_points-1:
                p.ds_to_next = math.sqrt((p.x - obs.waypoint_paths[lane_index][i+1].pos[0])**2 + (p.y - obs.waypoint_paths[lane_index][i+1].pos[1])**2)
                
                p.cur = (SMARTS_yawCorrect(obs.waypoint_paths[lane_index][i+1].heading, p.yaw) - p.yaw) / p.ds_to_next
            if i > 0:
                p.s = self.reference_line.points[-1].s + self.reference_line.points[-1].ds_to_next
            self.reference_line.points.append(p)
        
        lane_width = obs.waypoint_paths[0][0].lane_width
        self.field.lane_width = lane_width
        self.reference_line.points[-1].s= self.reference_line.points[-2].s
        self.curvature_smoother.smooth(self.reference_line)
        
        ego_heading = SMARTS_yawCorrect(obs.ego_vehicle_state.heading, self.reference_line.points[0].yaw)
        ego_frenet_heading = ego_heading - self.reference_line.points[0].yaw
        self.ego_vehicle.state.updatePose(x = obs.ego_vehicle_state.position[0], y = obs.ego_vehicle_state.position[1], yaw = ego_heading)
        vx = obs.ego_vehicle_state.linear_velocity[0] * math.cos(ego_heading) + obs.ego_vehicle_state.linear_velocity[1] * math.sin(ego_heading)
        vy = -obs.ego_vehicle_state.linear_velocity[0] * math.sin(ego_heading) + obs.ego_vehicle_state.linear_velocity[1] * math.cos(ego_heading)
        self.ego_vehicle.state.updateVelocity(vx = vx, vy = vy, avz = obs.ego_vehicle_state.yaw_rate)
        ego_vehicle_lane_index = obs.ego_vehicle_state.lane_index
        
        neighborhood_vehicle_grids = []
        for j in range(3): 
            temp_list = []
            for i in range(self.lane_num):
                temp_list.append([])
            neighborhood_vehicle_grids.append(temp_list)
        
        ego_edge_index = SMARTS_edgeIndex(obs.ego_vehicle_state.road_id)
            
        for neighborhood_vehicle in obs.neighborhood_vehicle_states:
            lat_index = neighborhood_vehicle.lane_index
            edge_index = SMARTS_edgeIndex(neighborhood_vehicle.road_id) 
            (is_too_far, relative_s) =  self.reference_line.xyToS(neighborhood_vehicle.position[0], neighborhood_vehicle.position[1])
            if is_too_far:
                continue
            if relative_s <= self.ego_grid_s_lenght  and relative_s >= 0:
                lon_index = 1
            elif relative_s > self.ego_grid_s_lenght and relative_s < self.ego_grid_s_lenght + self.perception_grid_s_lenght:
                lon_index = 2
            elif relative_s < 0 and relative_s > - self.perception_grid_s_lenght:
                lon_index = 0
            else:
                continue
            if relative_s >= 0:
                ref_heading = self.reference_line.sToYaw(relative_s)
            else:
                ref_heading=0

            neightbor_vehicle_heading = SMARTS_yawCorrect(neighborhood_vehicle.heading, ref_heading)
            delta_yaw = neightbor_vehicle_heading - ref_heading
            l = neighborhood_vehicle.lane_position.t + self.lane_center[lat_index] 
            vx = neighborhood_vehicle.speed
            #relative_s l delta_yaw v
            neighborhood_vehicle_grids[lon_index][lat_index].append((relative_s, l, delta_yaw, vx))
            
        ego_l = obs.ego_vehicle_state.lane_position.t + self.lane_center[obs.ego_vehicle_state.lane_index] 
        self.ego_vehicle.state.updateFrenetPose(0, ego_l, ego_frenet_heading)
        self.field.update(neighborhood_vehicle_grids)
        
        self.ego_vehicle.action.update(0, 0, 0)
        (_, current_dpf, frenet_direction, current_dpf_s, current_dpf_l, _) = \
            self.field.getIntensityAt(self.ego_vehicle.state.frenet_s, self.ego_vehicle.state.frenet_l,
                                      self.ego_vehicle.state.frenet_yaw, self.ego_vehicle.state.vx)
        
        if rl_a[0] >= 0:
            throttle=  rl_a[0]
            braking = 0
        else:
            braking = abs(rl_a[0])
            throttle= 0
        self.ego_vehicle.action.update(throttle, braking, rl_a[-1])
        (s,l,delta_yaw,vx) = self.ego_vehicle.frenet_kinetic_model_step(0.2, self.reference_line)
        (_, rl_pred_dpf, frenet_direction, rl_dpf_s, rl_dpf_l, _) = self.field.getIntensityAt(s,l,delta_yaw,vx)
        
        if human_a[0] >= 0:
            throttle=  human_a[0]
            braking = 0
        else:
            braking = abs(human_a[0])
            throttle= 0
        self.ego_vehicle.action.update(throttle, braking, human_a[-1])
        (s,l,delta_yaw,vx) = self.ego_vehicle.frenet_kinetic_model_step(0.2, self.reference_line)
        (_, human_pred_dpf, frenet_direction, human_dpf_s, human_dpf_l, _) = self.field.getIntensityAt(s,l,delta_yaw,vx)
        
        ################# FDPF Heatmap ####################### 
        # xlist = np.linspace(-20, 40, 100)
        # ylist = np.linspace(3.2*1.5, -3.2*1.5, 100)
        # X, Y = np.meshgrid(xlist, ylist)
        # intensity = np.zeros((100, 100))
        # intensity_with_bound = np.zeros((100, 100))
        # frenet_direction = np.zeros((100, 100))
        # intensity_s = np.zeros((100, 100))
        # intensity_l = np.zeros((100, 100))
        # intensity_bound = np.zeros((100, 100))
        # for i in range(100):
        #     for j in range(100):
        #         (intensity[i][j], intensity_with_bound[i][j], frenet_direction[i][j],
        #           intensity_s[i][j], intensity_l[i][j], intensity_bound[i][j]) = \
        #             self.field.getIntensityAt(X[i][j],Y[i][j],0,self.ego_vehicle.state.vx)

        # # fig=plt.figure()
        # # ax = plt.axes(projection='3d')
        # # cp = ax.contour3D(-Y, X,abs(intensity_l), 100)
        # # fig.colorbar(cp)
        
        # # fig, ax = plt.subplots(1,1)
        # # cp = ax.contourf(-Y, X, abs(intensity), cmap=cm.coolwarm)
        # # fig.colorbar(cp)
        
        # fig, ax = plt.subplots(1,1)
        # cp = ax.contourf(-Y, X, abs(intensity_with_bound), cmap=cm.coolwarm)
        # fig.colorbar(cp)
        
        # # fig, ax = plt.subplots(1,1)
        # # cp = ax.contourf(-Y, X, abs(intensity_bound), cmap=cm.coolwarm)
        # # fig.colorbar(cp)
        
        # plt.show(block=True)
        
        ############ Allocation ###########
        rl_dpf = rl_pred_dpf - current_dpf
        human_dpf = human_pred_dpf - current_dpf
        
        self.rl_authority, self.human_authority = self.coupled_normalize(abs(rl_dpf), abs(human_dpf))
        
        return self.rl_authority, self.human_authority
    
    def coupled_normalize(self, rl_risk, human_risk):
        if human_risk <= rl_risk or human_risk < 0.1:
            human_authority = 1.0
            rl_authority = 0.0
        else:
            authority = softmax(np.array([rl_risk*self.coeff, human_risk*self.coeff]))
            human_authority = authority[0]
            rl_authority = authority[-1]
            print (colored(str(rl_risk) + ' ' + str(human_risk) + ' ' +\
                           str(rl_authority) + ' ' + str(human_authority), 'green'))
        return rl_authority, human_authority
    
    def decoupled_normalize(self, s_risk, l_risk):
        
        if abs(s_risk[-1]) <= abs(s_risk[0]) or abs(s_risk[-1]) < 0.01:
            human_authority_s = 1.0
            rl_authority_s = 0.0
        else:
            authority = softmax(abs(l_risk) * self.coeff)
            rl_authority_s =  authority[-1]
            human_authority_s = authority[0]
            print (colored(str(rl_authority_s) + ' ' + str(human_authority_s), 'red'))
        
        if abs(l_risk[-1]) <= abs(l_risk[0]) or abs(l_risk[-1]) < 0.01:
            human_authority_l = 1.0
            rl_authority_l = 0.0
        else:
            authority = softmax(abs(s_risk) * self.coeff)
            rl_authority_l =  authority[-1]
            human_authority_l = authority[0]
            print (colored(str(rl_authority_l) + ' ' + str(human_authority_l), 'green'))
        return np.array([rl_authority_s, rl_authority_l]), np.array([human_authority_s, human_authority_l])
        