import math
from tomlkit import string

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

curve_point = []
end_yaw = 0
resolution_distance = 2
R = 80
resolution_angle = resolution_distance / R

center_x = 100
center_y = R
yaw = -math.pi / 2

while yaw < end_yaw:
    yaw += resolution_angle
    point = Point(x=center_x + math.cos(yaw) * R, y=center_y + math.sin(yaw) * R)
    curve_point.append(point)

result = ""
for p in curve_point:
    result += str(round(p.x,2)) + "," + str(round(p.y,2)) + " "

print(result)