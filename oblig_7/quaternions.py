#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:19:16 2025

@author: chcurry
"""

import numpy as np


class Quaternion:
    def __init__(self, scalar, vec):
        # scalar 1xn, vec 3xn
        s = np.array(scalar)
        v = np.array(vec)
        if (np.size(v, axis=0) != 3) or s.ndim != 1:
            raise ValueError(
                "Cannot initialize quaternion with fewer or more than four parameters"
            )
        if np.size(v, axis=1) != s.size:
            raise ValueError("Number of scalars and vectors not equal")
        self.scalar = s
        self.vec = v

    def __repr__(self):
        # this could be written better...
        return f"quaternion {self.scalar} + i*{self.vec[0,:]} + j*{self.vec[1,:]} + k*{self.vec[2,:]}"

    def __add__(self, other):
        # adding two Quaternion instances "self" and "other"
        # ***********************************************
        scalar = self.scalar + other.scalar
        vec = self.vec + other.vec
        return Quaternion(scalar, vec)  # add/adapt code
        # ***********************************************

    def __mul__(self, other):
        # # multiplication of two quaterions, note that we do not implement
        # # the product scalar * Quaternion (which could be done as well)
        # # ***********************************************
        # scalar = self.scalar * other.scalar - np.sum(self.vec * other.vec, axis=0)
        # vec = (
        #     self.scalar * other.vec
        #     + other.scalar * self.vec
        #     + np.cross(self.vec, other.vec, axis=0)
        # )
        # return Quaternion(scalar, vec)  # add/adapt code

        # def __mul__(self, other):
        s1, v1 = self.scalar, self.vec
        s2, v2 = other.scalar, other.vec
        new_scalar = s1 * s2 - np.dot(v1, v2)
        new_vec = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return Quaternion(new_scalar, new_vec)
        # ***********************************************

    def __abs__(self):
        # return the length of a quaternion
        # ***********************************************
        normsq = self.scalar * self.scalar + np.sum(self.vec * self.vec, axis=0)
        return np.sqrt(normsq)
        # ***********************************************

    def inverse(self):
        # return the inverse quaternion; there is no need to exclude (0,[0,0,0])
        # for this task but in a bigger project, that would be good practice
        # ***********************************************
        scalar = self.scalar / abs(self)
        vec = self.vec / -abs(self)
        return Quaternion(scalar, vec)  # add/adapt code
        # ***********************************************

    def scale(self, a):
        self.scalar = a * self.scalar
        self.vec = a * self.vec
        return Quaternion(self.scalar, self.vec)

    @classmethod
    def from_rotation(cls, vector, angle):
        # This is an alternative instantiator for Quaternions: From a given
        # vector and angle the quaternion is created that rotates around the
        # vector by that angle
        return cls(
            [np.cos(angle / 2)], np.transpose([np.sin(angle / 2) * np.array(vector)])
        )

    def rotate_vector(self, vector):
        # Use the quaternion to rotate a given vector by the rotation
        # represented by that quaternion. Again, we do not require "safety nets"
        n = np.size(vector, axis=1)
        versor = Quaternion(np.zeros(n), vector)
        # ***********************************************
        versor1 = self * versor * self.inverse()
        return versor1.vec  # add/adapt code
        # ***********************************************
