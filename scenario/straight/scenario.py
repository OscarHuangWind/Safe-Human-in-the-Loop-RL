#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:41:48 2023

@author: oscar
"""

import random
from pathlib import Path

from smarts.core import seed
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Mission, EndlessMission, Route, TrafficActor
from smarts.sstudio.types import Distribution, LaneChangingModel, JunctionModel

seed(10)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=(route[0], random.randint(0, 2), 'random'),
                end=(route[1], random.randint(0, 2), "max"),
            ),
            repeat_route=True,
            rate=1,
            end=10,  # `rate=1` adds 1 additional vehicle per hour. So set `end` < 1*60*60 secs to avoid addition of more vehicles after the initial flow. This prevents traffic congestion.
            actors={
                t.TrafficActor(
                    name="car",
                    max_speed = 8,
                    depart_speed = 'max',
                    lane_changing_model = LaneChangingModel(impatience=1, cooperative=0.25, pushy=1.0),
                    speed=Distribution(mean=0.5, sigma=0.8),
                    vehicle_type="passenger"
                    # vehicle_type=random.choice(
                    #     ["passenger", "coach", "bus", "trailer", "truck"]
                    # ),
                ): 1
            },
        )
        for route in [("E0", "E0")] * 13
    ]
)

ego_missions = [
    EndlessMission(
        begin=["E0", 1, 1],
    )
]

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
