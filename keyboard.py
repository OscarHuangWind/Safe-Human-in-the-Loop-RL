#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:28:29 2023

@author: oscar
"""

"""
This examples runs the human-keyboard Agent, which allows you to control and monitor input devices.
"""

from time import sleep

try:
    from pynput.keyboard import Key, KeyCode, Listener
except ImportError:
    raise ImportError("pynput dependency is missing, please pip install -e .[extras]")

from smarts.core.agent import Agent

class HumanKeyboardAgent(Agent):
    def __init__(self):
        # initialize the keyboard listener
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

        # Parameters for the human-keyboard agent
        # you need to change them to suit the scenario
        # These values work the best with highway

        self.INC_THROT = 0.1 
        self.INC_STEER = 0.04

        self.MAX_THROTTLE = 0.6
        self.MIN_THROTTLE = 0.15

        self.MAX_BRAKE = 1.0
        self.MIN_BRAKE = 0.0

        self.MAX_STEERING = 1.0
        self.MIN_STEERING = -1.0

        self.THROTTLE_DISCOUNTING = 0.99
        self.BRAKE_DISCOUNTING = 0.95
        self.STEERING_DISCOUNTING = 0.9

        # initial values
        self.steering_angle = 0.0
        self.throttle = 0.48
        self.brake = 0.0
        
        self.intervention = False
        self.slow_down = False

    def on_press(self, key):
        """To control, use the keys:
        Up: to increase the throttle
        Left Alt: to increase the brake
        Left: to decrease the steering angle
        Right: to increase the steering angle
        """
        if key == Key.up:
            self.throttle = min(self.throttle + self.INC_THROT, self.MAX_THROTTLE)
            self.brake = 0.0
            self.intervention = True
        elif key == Key.down:
            self.throttle = 0.0
            self.brake = min(self.brake + 10.0 * self.INC_THROT, self.MAX_BRAKE)
            self.intervention = True
        elif key == Key.right:
            self.steering_angle = min(
                self.steering_angle + self.INC_STEER, self.MAX_STEERING
            )
            self.intervention = True
        elif key == Key.left:
            self.steering_angle = max(
                self.steering_angle - self.INC_STEER, self.MIN_STEERING
            )
            self.intervention = True
        elif key == Key.ctrl:
            self.intervention = True
            self.throttle = 0.15
            self.brake = 0.0
            self.steering_angle = 0.0
        elif key == Key.space:
            self.intervention = True
            self.throttle =  0.0
            self.brake = 0.2
            self.steering_angle = 0.0
        elif key == Key.enter:
            self.throttle =  0.0
            self.brake = 0.0
            self.steering_angle = 0.0
            self.intervention = False
        elif key == Key.shift:
            self.slow_down = True
        elif key == Key.tab:            
            self.slow_down = False


    def act(self):
        # discounting ..
        sleep(1/40)
        self.throttle = max(
            self.throttle * self.THROTTLE_DISCOUNTING, self.MIN_THROTTLE
        )
        self.steering_angle = self.steering_angle * self.STEERING_DISCOUNTING
        self.brake = self.brake * self.BRAKE_DISCOUNTING
        # send the action
        self.action = [self.throttle, self.brake, self.steering_angle]
        return self.action
