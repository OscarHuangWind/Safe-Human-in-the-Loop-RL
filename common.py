#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:22:16 2023

@author: oscar
"""

import math
import yaml
import numpy as np

def SMARTS_yawCorrect(yaw, ref_yaw):
    yaw += math.pi / 2

    return yaw
        
def SMARTS_edgeIndex(road_id):
    if road_id == 'E0':
        return 0
    elif road_id == 'E1':
        return 1
    elif road_id == 'E2':
        return 2
    else:
        print("Road_id is error, it is " + road_id)

def linearInterpolation(a, b, rate_of_a):
    return a * rate_of_a + b * (1 - rate_of_a)

# trajectory common
class TrajPoint:
    def __init__(self, x=0, y=0, yaw=0, cur=0, 
                       frenet_l=0, frenet_s=0, delta_yaw=0):
        # Cartesian Info 
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cur = cur
        self.s = 0
        self.ds_to_next = 0

        # Temporal Info 
        self.vx = 0
        self.ax = 0
        self.t = 0

        # Frenet Info 
        self.frenet_l = frenet_l
        self.frenet_s = frenet_s
        self.delta_yaw = delta_yaw
        
        # Differentiation of Distance 
        self.ax_dot = 0 
        self.cur_dot = 0 

        self.l_min = 0
        self.l_max = 0
        self.v_min = 0
        self.v_max = 100

    def setFrenetLCorridor(self, center, half_width=0): # Spatio-temporal Corridor Boundary 
        self.l_min = center - half_width
        self.l_max = center + half_width

    def setVxCorridor(self, center, half_width=0): # Spatio-temporal Corridor Boundary
        self.v_min = center - half_width
        self.v_max = center + half_width

    def setVxMinMax(self, input_min, input_max):
        self.v_min = input_min
        self.v_max = input_max

class Trajectory:
    def __init__(self):
        self.points = []
        # self.points = queue.Queue()

    def rotate(self, theta):
        for i in range(0, len(self.points)):
            delta_x = self.points[i].x - self.points[0].x
            delta_y = self.points[i].y - self.points[0].y
            self.points[i].x = self.points[0].x + delta_x * math.cos(theta) - delta_y * math.sin(theta)
            self.points[i].y = self.points[0].y + delta_x * math.sin(theta) + delta_y * math.cos(theta)
            self.points[i].yaw = self.points[i].yaw + theta

    def translate(self, x = 0, y = 0):
        for i in range(0, len(self.points)):
            self.points[i].x += x
            self.points[i].y += y

    def calculateS(self):
        self.points[0].s = 0
        self.points[0].ds_to_next = math.sqrt((self.points[1].x - self.points[0].x)**2 + (self.points[1].y - self.points[0].y)**2)
        for i in range(1,len(self.points) - 1):
            self.points[i].s = self.points[i - 1].s + self.points[i - 1].ds_to_next
            self.points[i].ds_to_next = math.sqrt((self.points[i + 1].x - self.points[i].x)**2 + (self.points[i + 1].y - self.points[i].y)**2)
        self.points[-1].s = self.points[-2].s + self.points[-2].ds_to_next
        self.points[-1].ds_to_next = 0

    def calculateCartesianInfo(self, reference_line):
        for i in range(len(self.points)):
            self.points[i].x = float(reference_line.points[i].x + self.points[i].frenet_l * math.cos(reference_line.points[i].yaw + math.pi / 2))
            self.points[i].y = float(reference_line.points[i].y + self.points[i].frenet_l * math.sin(reference_line.points[i].yaw + math.pi / 2))
            self.points[i].yaw = float(reference_line.points[i].yaw + self.points[i].delta_yaw)

            self.points[i].frenet_s = float(self.points[i].s)

            self.points[i].l_max = float(reference_line.points[i].l_max - self.points[i].frenet_l)
            self.points[i].l_min = float(reference_line.points[i].l_min - self.points[i].frenet_l)
        self.calculateS()

    def sToYaw(self, s):
        if s > self.points[-1].s:
            print("Inputted s should smaller or equal to the final s of reference line.")
            print("s is "+ str(s))
            print("ref max s is "+ str(self.points[-1].s))
            return
        elif s < 0:
            print("Inputted s should bigger than 0.")
            print("s is "+ str(s))
            return
            
        for i in range(len(self.points)):
            if self.points[i].s == s:
                return self.points[i].yaw
            elif self.points[i].s > s:
                rate = (self.points[i].s - s) / self.points[i - 1].ds_to_next
                return linearInterpolation(self.points[i - 1].yaw, self.points[i].yaw,rate)
            
    def sToCur(self, s):
        if s > self.points[-1].s:
            print("Inputted s should smaller or equal to the final s of reference line.")
            return
        for i in range(len(self.points)):
            if self.points[i].s == s:
                return self.points[i].cur
            elif self.points[i].s > s:
                rate = (self.points[i].s - s) / self.points[i - 1].ds_to_next
                return linearInterpolation(self.points[i - 1].cur, self.points[i].cur,rate)
            
    def xyToS(self, x, y, resolution = 0.1, back_extension_distance = 10):
        min_distance = 999
        result_s = 0
        
        s = -resolution
        while s > -back_extension_distance:
            ref_x = self.points[0].x + math.cos(self.points[0].yaw) * s
            ref_y = self.points[0].y + math.sin(self.points[0].yaw) * s
            s -= resolution
            distance =  math.sqrt((ref_x - x)**2 + (ref_y - y)**2)
            if distance < min_distance:
                min_distance = distance
                result_s = s
            
        for i in range(len(self.points)-1):
            s = self.points[i].s
            while s < self.points[i + 1].s:
                ref_x = self.points[i].x + math.cos(self.points[i].yaw) * (s - self.points[i].s)
                ref_y = self.points[i].y + math.sin(self.points[i].yaw) * (s - self.points[i].s)
                distance =  math.sqrt((ref_x - x)**2 + (ref_y - y)**2)
                if distance < min_distance:
                    min_distance = distance
                    result_s = s
                s += resolution
            
        is_too_far = False
        if min_distance > 10:
            is_too_far = True
        return (is_too_far, result_s)
    
import casadi as ca
class CurvatureSmoother():
    def __init__(self,N, is_debuging=False):
        self.N = N

        self.initVariables()
        self.setCosts()
        self.setConstraints()
        self.constructSolver()

    def initVariables(self):
        # State Variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.kappa = ca.SX.sym('kappa') 
        self.states = ca.vertcat(self.x, self.y, self.theta, self.kappa)
        self.n_states = self.states.size()[0]
        self.X = ca.SX.sym('X', self.n_states, self.N+1)
        # Control Variables 
        self.kappa_dot = ca.SX.sym('kappa_dot') 
        self.controls = ca.vertcat(self.kappa_dot)
        self.n_controls = self.controls.size()[0] 
        self.U = ca.SX.sym('U', self.n_controls, self.N) 
        self.optimization_variables = ca.vertcat(ca.reshape(self.U, -1, 1), ca.reshape(self.X, -1, 1))
        # # Parameter Variables
        self.xy = ca.SX.sym('initial_xy', 2, self.N + 1)
        self.ds = ca.SX.sym('ds', self.N, 1)
        self.auxiliary_variables = ca.vertcat(ca.reshape(self.xy, -1, 1), ca.reshape(self.ds, -1, 1))

    def setCosts(self):
        self.cost = 0
        for i in range(self.N):
            self.cost += (self.X[0,i] - self.xy[0, i])**2 + (self.X[1,i] - self.xy[1, i])**2
            self.cost += 50 * self.U[0,i]**2

    def setConstraint(self, constraint, limit_low, limit_up):
        if limit_low > limit_up:
            print("setConstraint: limit_low is bigger than limit_up.")
            return
        self.constraints.append(constraint)
        self.constraint_limits_low.append(limit_low)
        self.constraint_limits_up.append(limit_up)

    def setConstraints(self):
        self.constraints = [];self.constraint_limits_low = [];self.constraint_limits_up = []
        self.variable_limits_low = [];self.variable_limits_up = []
        self.setConstraint(self.X[0, 0] - self.xy[0, 0], 0, 0)
        self.setConstraint(self.X[1, 0] - self.xy[1, 0], 0, 0)
        rhs = ca.vertcat(ca.cos(self.theta), ca.sin(self.theta), self.kappa, self.kappa_dot)
        f = ca.Function('f', [self.states, self.controls], [rhs], ['input_state', 'control_input'], ['rhs'])
        for i in range(self.N):
            st = self.X[:,i]
            f_value = f(self.X[:, i], self.U[:, i]) 
            st_next = self.X[:, i + 1]
            st_next_euler = st + self.ds[i] * f_value
            for j in range(self.n_states):
                self.setConstraint(st_next[j] - st_next_euler[j], 0, 0)
        for _ in range(self.N):
            self.variable_limits_low.append( - ca.inf )
            self.variable_limits_up.append(   ca.inf )
        for _ in range(self.N+1):
            self.variable_limits_low.append(-ca.inf)
            self.variable_limits_up.append(  ca.inf)
            self.variable_limits_low.append(-ca.inf)
            self.variable_limits_up.append(  ca.inf)
            self.variable_limits_low.append(-ca.inf)
            self.variable_limits_up.append(  ca.inf)
            self.variable_limits_low.append( - 0.5 )
            self.variable_limits_up.append(    0.5 )

    def constructSolver(self):
        nlp_prob = {'f': self.cost, 'x': self.optimization_variables, 'p':self.auxiliary_variables, 'g':ca.vertcat(*self.constraints)}
        opts_setting = {'ipopt.max_iter':1000, 'ipopt.print_level':0, 'print_time':False, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-10}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    def smooth(self, input_path):
        init_control = np.zeros(self.N * self.n_controls + (self.N + 1) * self.n_states)#热启动初始数值
        for i in range(self.N):
            init_control[i * self.n_controls + 0] = 0.00 # cur_dot
        for i in range(self.N + 1):
            init_control[self.N * self.n_controls + self.n_states * i + 0] = input_path.points[i].x # x
            init_control[self.N * self.n_controls + self.n_states * i + 1] = input_path.points[i].y # y
            init_control[self.N * self.n_controls + self.n_states * i + 2] = input_path.points[i].yaw # yaw
            init_control[self.N * self.n_controls + self.n_states * i + 3] = input_path.points[i].cur # cur
        xy = []
        ds = []
        for i in range(len(input_path.points) - 1):
            ds.append(input_path.points[i].ds_to_next)
        for p in input_path.points:
            xy.append(p.x)
            xy.append(p.y)
        ds = np.array(ds).reshape(-1, 1)
        xy = np.array(xy).reshape(-1, 1)
        parameter = np.concatenate((xy, ds))
        result = self.solver(x0 = init_control, p = parameter, 
                            lbg = self.constraint_limits_low, lbx = self.variable_limits_low, 
                            ubg = self.constraint_limits_up,  ubx = self.variable_limits_up)
        for i in range(self.N + 1):
            input_path.points[i].yaw = float(result['x'][self.n_controls * self.N + self.n_states * i + 2])
            input_path.points[i].cur = float(result['x'][self.n_controls * self.N + self.n_states * i + 3])
            if input_path.points[i].yaw < 0.1 / 180 * math.pi:
                input_path.points[i].yaw = 0
            if input_path.points[i].cur < 0.0001:
                input_path.points[i].cur = 0
        if  len(input_path.points) > self.N:
            for i in range(self.N,len(input_path.points)):
                input_path.points[i].cur = input_path.points[self.N-1].cur
                input_path.points[i].yaw = input_path.points[self.N-1].yaw

class VehicleState():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.yaw = 0

        self.vx = 0
        self.vy = 0
        self.avz = 0

        self.ax = 0
        self.ay = 0
        self.aavz = 0
        
        self.frenet_s = 0
        self.frenet_l = 0
        self.frenet_yaw = 0

    def updatePose(self, x=0, y=0, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        
    def updateFrenetPose(self, s=0, l=0, frenet_yaw=0):
        self.frenet_s = s
        self.frenet_l = l
        self.frenet_yaw = frenet_yaw

    def updateVelocity(self, vx=0, vy=0, avz=0):
        self.vx = vx
        self.vy = vy
        self.avz = avz

    def updateAcceleration(self, ax=0, ay=0, aavz=0):
        self.ax = ax
        self.ay = ay
        self.aavz = aavz

    def getVehicleSlipAngle(self):
        return math.atan2(self.vy, self.vx)

class VehicleAction():
    def __init__(self):
        self.steering = 0
        self.throttle = 0
        self.braking = 0

    def update(self, throttle, braking, steering):
        self.steering = steering
        self.throttle = throttle
        self.braking = braking

class VehicleParam():
    def __init__(self, config_file):
        print(config_file)
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.L = config['L']
            self.Lf = config['Lf']
            self.Lr = config['Lr']
            self.W = config['W']
            self.width = config['width']
            self.length = config['length']

            self.min_radius = config['Minimum turning radius']

            self.steering_coefficient = config['max_steering'] / config['steering_gear_ratio']
            self.throttle_coefficient = 4 * config['max_torque'] / config['wheel_radius'] / config['mass']
            self.braking_coefficient = 4 * config['max_btorque'] / config['wheel_radius'] / config['mass']
            self.ct_action = config['Calibrated Throttle Action']
            self.ct_acceleration = config['Calibrated Throttle Acceleration']
            self.cb_action = config['Calibrated Braking Action']
            self.cb_acceleration = config['Calibrated Braking Acceleration']

class Vehicle():
    def __init__(self, config_file):
        self.state = VehicleState() 
        self.param = VehicleParam(config_file)

    def steeringToSteeringAngle(self, steering):
        return - self.param.steering_coefficient * steering
    
    def throttleToAcceleration(self, throttle, Table = False):
        if Table:
            for i in range(len(self.param.ct_action)):
                if throttle == self.param.ct_action[i]:
                    return self.param.ct_acceleration[i]
                elif throttle < self.param.ct_action[i]:
                    a = (throttle - self.param.ct_action[i - 1]) / (self.param.ct_action[i] - self.param.ct_action[i - 1])
                    return linearInterpolation(self.param.ct_acceleration[i-1], self.param.ct_acceleration[i], (1 - a))
        else:
            return throttle * self.param.throttle_coefficient
            
    def brakingToAcceleration(self, braking, Table = False):
        if Table:
            for i in range(len(self.param.cb_action)):
                if braking == self.param.cb_action[i]:
                    return self.param.cb_acceleration[i]
                elif braking < self.param.cb_action[i]:
                    a = (braking - self.param.cb_action[i - 1]) / (self.param.cb_action[i] - self.param.cb_action[i - 1])
                    return linearInterpolation(self.param.cb_acceleration[i-1], self.param.cb_acceleration[i], (1 - a))
        else:
            return braking * self.param.braking_coefficient
        
    def kinetic_model_step(self, dt, sim_resolution_dt = 0.01):
        yaw_rate = self.steeringToSteeringAngle(self.action.steering)
        longitudinal_acceleration = self.throttleToAcceleration(self.action.throttle) - self.brakingToAcceleration(self.action.braking)
        next_x = self.state.x
        next_y = self.state.y
        next_yaw = self.state.yaw
        next_vx = self.state.vx
        for i in range(int(dt / sim_resolution_dt)):
            next_x += sim_resolution_dt * math.cos(next_yaw) * next_vx
            next_y += sim_resolution_dt * math.sin(next_yaw) * next_vx
            next_yaw += sim_resolution_dt * yaw_rate
            next_vx += sim_resolution_dt * longitudinal_acceleration
        return (next_x, next_y, next_yaw, next_vx)
    
    def frenet_kinetic_model_step(self, dt, reference_line, sim_resolution_dt = 0.01):
        yaw_rate = self.steeringToSteeringAngle(self.action.steering)
        longitudinal_acceleration = self.throttleToAcceleration(self.action.throttle) - self.brakingToAcceleration(self.action.braking)
        next_s = self.state.frenet_s
        next_l = self.state.frenet_l
        next_delta_yaw = self.state.frenet_yaw
        next_vx = self.state.vx
        for i in range(int(dt / sim_resolution_dt)):
            next_s += max(sim_resolution_dt * math.cos(next_delta_yaw) * next_vx, 0)
            next_l += sim_resolution_dt * math.sin(next_delta_yaw) * next_vx
            next_delta_yaw += sim_resolution_dt * yaw_rate - sim_resolution_dt * next_vx * reference_line.sToCur(next_s)
            next_vx += sim_resolution_dt * longitudinal_acceleration
        return (next_s, next_l, next_delta_yaw, next_vx)
