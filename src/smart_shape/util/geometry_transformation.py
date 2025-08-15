# -*- coding: utf-8 -*-
import numpy as np

def rotatePoint(centerPoint, point, angle):
    """

    :param centerPoint:
    :param point:
    :param angle:
    :return:
    Rotates a point around another centerPoint. Angle is in degrees.

    Rotation is counter-clockwise
    """

    import math

    angle = math.radians(angle)
    temp_point = point[0] - centerPoint[0], point[1] - centerPoint[1]
    temp_point = (
        temp_point[0] * math.cos(angle) - temp_point[1] * math.sin(angle),
        temp_point[0] * math.sin(angle) + temp_point[1] * math.cos(angle),
    )
    temp_point = temp_point[0] + centerPoint[0], temp_point[1] + centerPoint[1]
    return temp_point


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # parallel lines
        return None
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def rotate_multiple_points(p, origin=(0, 0), degrees=0):
    """
    points=[(200, 300), (100, 300)]
    origin=(100,100)

    new_points = rotate(points, origin=origin, degrees=10)
    print(new_points)
    """

    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    v2_countercross_wrt_v1 = (v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0]) < 0
    if v2_countercross_wrt_v1:
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        return -np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
