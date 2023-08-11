#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:04:15 2021

@author: nvilla
"""
import numpy as np
import scipy.spatial as sp
import mpc_interface.tools as use
from enum import Enum

# # TODO: make some visualization of the constraints graphically

## TODO IDEA: don't use enums, pass as string, or in csae, put enum inside the class
class SPACE(Enum):
    TS = 1
    SS = 2    

class Constraint:
    def __init__(
        self,
        variable_name,
        extreme,
        axes=None,
        arrow=None,
        center=None,
        L=None,
        schedule=None,
    ):
        """
        ARGUMENTS:

            variable_out : dict, {name: String, combination: dict }

            axes : list of strings ex: ["_x", "_y", "_z"] always starting "_"

            arrow :   ndarray with shape [m, len(axes)] or list len(axes)

            extreme : ndarray with shape [m, 1] or single number, or list.

            center :  ndarray with shape [m, len(axes)] or list len(axes)

            L : [m, dim] x len(axes)

            schedule : ndarray with shape [dim] of bool -> Selecting t values

        The null value of L is an empty list [] and the null value of
        schedule is range(0). These values can be used in the update
        function to remove them.

        Each instance of this class provides instructions to generate
        m (or t (or horizon_lenght)) lineal constraints of the form:

            arrow * ( V - center ) < extreme

        where V is:

            V = [  Lx @ v_x[schedule], Ly @ v_y[schedule], ... ]

        based on the variable_in v = [v_x, v_y] which is defined by 'variable' and 'axes'.
        """
        self.variable_name = variable_name 
        self.axes = [""] if axes is None else axes
        if not isinstance(self.axes, list):
            raise TypeError("The axes must be a list of string")
        self.axes_len = len(self.axes)

        self.schedule = range(0) if schedule is None else schedule

        if L is None:
            self.L = []
        else:
            self.__arrange_L(L)

        self.extreme = np.array(extreme).reshape([-1, 1]).astype(float)
        self.__initialize_geometry(arrow, center)
        self.__check_geometry()
        self.__normalize()

    def __arrange_L(self, L):
        """This function requires an updated schedule."""
        self.L = L if isinstance(L, list) else [L]

        if len(self.L) == 1:
            self.L = self.L * self.axes_len
        elif len(self.L) not in (self.axes_len, 0):
            raise IndexError(
                "'L' must have 0, 1 or len(axes) = {} elements".format(self.axes_len)
            )

        for i, sl in enumerate(self.L):
            self.L[i] = sl if len(L.shape)-1 else sl[None, :]
            if self.schedule and sl.shape[-1] != self.t:
                raise ValueError(
                "arrays in L must have {} columns, which is given by the 'schedule'.".format(self.t)
                )

    #    if self.schedule and np.any([sl.shape[-1] != self.t for sl in self.L]):
    #        raise ValueError(
    #            "arrays in L must have {} ".format(self.t)
    #            + "columns, which is given by the 'schedule'."
    #        )

    def __initialize_geometry(self, arrow, center):
        if arrow is None:
            if self.axes_len == 1:
                self.arrow = np.ones(self.extreme.shape)

            else:
                raise ValueError(
                    "When using multiple axes, "
                    + "some normal direction 'arrow' must be provided"
                )

        else:
            self.arrow = np.array(arrow).reshape([-1, self.axes_len])

        if center is None:
            self.center = np.zeros([1, self.axes_len])

        else:
            self.center = np.array(center).reshape([-1, self.axes_len])

    def __check_geometry(self):
        rows = np.array(
            [self.arrow.shape[0], self.center.shape[0], self.extreme.shape[0]]
        )
        rows_not_1 = rows[rows != 1]

        if np.any(rows_not_1) and np.any(rows_not_1 != rows_not_1[0]):
            raise ValueError(
                "The number of rows in 'arrow', 'center' and "
                + "'extreme' must be equal or 1, but they are "
                + "{} respectively".format(rows)
            )
        if self.axes_len > 1:
            cols = np.array([self.arrow.shape[1], self.center.shape[1]])
            if np.any(cols != self.axes_len):
                raise IndexError(
                    (
                        "'arrow' and 'center' have {} columns "
                        + "but they must have {}, one per "
                        + "axis."
                    ).format(cols, self.axes_len)
                )

    def __normalize(self):
        if self.arrow.shape[0] != self.extreme.shape[0]:
            if self.arrow.shape[0] == 1:
                self.arrow = np.resize(
                    self.arrow, [self.extreme.shape[0], self.axes_len]
                )
            elif self.extreme.shape[0] == 1:
                self.extreme = np.resize(self.extreme, [self.arrow.shape[0], 1])

        for i, extreme in enumerate(self.extreme):
            # We adapt the arrow to have always positive extreme
            if extreme < 0:
                self.extreme[i] = -self.extreme[i]
                self.arrow[i] = -self.arrow[i]

        # Is there a good reason to make unit vectors in arrow?

    @property
    def m(self):
        if self.L:
            return self.L[0].shape[0]
        return None

    @property
    def t(self):
        if self.schedule:
            return self.schedule.stop - self.schedule.start
        return None

    @property
    def nlines(self):
        if self.L:
            return self.m

        if self.schedule:
            return self.t

        sizes = np.array(
            [self.arrow.shape[0], self.center.shape[0], self.extreme.shape[0]]
        )
        if np.all(sizes == 1):
            return None

        sizes_not_1 = sizes[sizes != 1]
        return sizes_not_1[0]

    def matrices(self):
        if self.L:
            return [self.arrow[:, i][:, None] * l for i, l in enumerate(self.L)]
        else:
            return [self.arrow[:, i][:, None] for i in range(self.axes_len)]

    def bound(self):
        return self.extreme + np.sum(self.arrow * self.center, axis=1).reshape([-1, 1])

    def broadcast(self):
        if self.nlines():
            if self.extreme.shape[0] == 1:
                self.extreme = np.resize(self.extreme, [self.nlines, 1])

            if self.arrow.shape[0] == 1:
                self.arrow = np.resize(self.arrow, [self.nlines, self.axes_len])

            if self.center.shape[0] == 1:
                self.center = np.resize(self.center, [self.nlines, self.axes_len])

    def __extend(self, element, N, mode = "edge"):
        n = N - element.shape[0]
        return np.pad(element, ((0,n),(0,0)), mode)

    def __shrink(self, element, N):
        return element[:N, :]

    def forecast(self, N):
        if self.m or self.t:
            ##TODO: adapt for the case when m or t are not None.
            # In this case we could vstack the matrix L according to "N"
            # Or introduce options to extend or shrink L in different ways.
            if not (N == self.nlines or N == 1):
                raise IndexError(
                    (
                        "The horizon lenght in this constraint was set to {} "
                        + "by L or the schedule, it is not possible to forecast "
                        + "it to {}."
                    ).format(self.nlines, N)
                )

        self.arrow = self.__extend(self.arrow, N) if N > self.arrow.shape[0] else self.__shrink(self.arrow, N)
        self.center = self.__extend(self.center, N) if N > self.center.shape[0] else self.__shrink(self.center, N)
        self.extreme = self.__extend(self.extreme, N) if N > self.extreme.shape[0] else self.__shrink(self.extreme, N)

    def update(self, extreme=None, arrow=None, center=None, L=None, schedule=None):
        if schedule is not None:
            self.schedule = schedule

        if L is not None:
            self.__arrange_L(L)

        if extreme is not None:
            self.extreme = np.array(extreme).reshape([-1, 1])
        if arrow is not None:
            self.arrow = np.array(arrow).reshape([-1, self.axes_len])
        if center is not None:
            self.center = np.array(center).reshape([-1, self.axes_len])

        if extreme is not None or arrow is not None:
            self.__check_geometry()
            self.__normalize()
        elif center is not None:
            self.__check_geometry()

    def is_feasible(self, points, space="SS"):
        """Introduce a list of points to verify the feasibility of each
        one separately, or introduce a np.vstack of row points to check
        feasibility along the time horizon. Or a list of np.stacks if
        prefered.
        for evaluations along the horizon, points are considered to ve at the
        scheduled times.
        """
        if isinstance(points, list):
            if space == "SS":
                return [self._is_feasible_SS(point) for point in points]
            elif space == "TS":
                return [self._is_feasible_TS(point) for point in points]
        else:
            if space == "SS":
                return self._is_feasible_SS(points)
            elif space == "TS":
                return self._is_feasible_TS(points)

    def SS_to_TS(self, ss_point):
        """The ss_point must be arranged with one column per TS_axis.
        ex: ss_point = [p_x, p_y, p_z] = [[pv_x, pv_y, pv_z],
                                          [pw_x, pw_y, pw_z],]
        where x, y and z are TS axes, and v and w are SS axes
        """
        if self.L:
            return np.vstack(
                [(ll @ p.T) for ll, p in zip(self.L, np.transpose(ss_point))]
            ).T
        return ss_point

    def _is_feasible_SS(self, ss_point):
        ts_point = self.SS_to_TS(ss_point)
        return self._is_feasible_TS(ts_point)

    def _is_feasible_TS(self, ts_point):
        return np.sum(self.arrow * (ts_point - self.center), axis=1) < self.extreme

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.axes != [""]:
            axes = "_" + "".join(axis[1:] for axis in self.axes)
        else:
            axes = ""

        text = "\n\tvariable name: " + self.variable_name + axes
        text += (
            "\nwith L = " + str(",\n".join(str(ll) for ll in self.L))
            if self.L != []
            else ""
        )
        text += "\n\t\tarrow: " + ("\n\t\t"+" " * 7).join(
            str(arrow) for arrow in self.arrow
        )
        text += "\n\t\tcenter: " + ("\n\t\t"+" " * 8).join(
            str(center) for center in self.center
        )
        text += "\n\t\textreme: " + ("\n\t\t"+" " * 9).join(
            str(extreme) for extreme in self.extreme
        )
        text += "\n"
        return text


class Box:
    def __init__(self, time_variant=None, how_to_update=None):

        self.constraints = []

        self.time_variant = time_variant
        self.naxes = 0

        if how_to_update is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update

    @classmethod
    def task_space(
        cls,
        variable_name,
        vertices,
        axes=None,
        L=None,
        schedule=None,
        time_variant=None,
        how_to_update=None,
    ):
        box = cls(time_variant, how_to_update)
        axes = [""] if axes is None else axes
        if not isinstance(axes, list): #CHECK
            raise TypeError("The axes must be a list of string")
        box.naxes = len(axes)

        arrows, extremes, center = box_boundaries(vertices)
        for i, extreme in enumerate(extremes):
            box.constraints.append(
                Constraint(variable_name, extreme, axes, arrows[i], center, L, schedule)
            )
        return box

    @classmethod
    def state_space(
        cls,
        variable_name,
        vertices,
        axes=None,
        schedule=None,
        time_variant=None,
        how_to_update=None,
    ):

        box = cls(time_variant, how_to_update)
        axes = [""] if axes is None else axes
        if not isinstance(axes, list): #CHECK
            raise TypeError("The axes must be a list of string")
        box.naxes = len(axes)

        arrows_SS, extremes_SS, center_SS = box_boundaries(vertices)
        center = np.sum(arrows_SS * center_SS, axis=1).reshape([-1, 1])

        for i, extreme in enumerate(extremes_SS):
            box.constraints.append(
                Constraint(
                    variable_name,
                    extreme,
                    axes=axes,
                    center=center[i],
                    L=arrows_SS[i],
                    schedule=schedule,
                )
            )
        return box

    def forecast(self, cast):
        for boundary in self.constraints:
            boundary.forecast(cast)

    @classmethod
    def task_space_exterior(
        cls,
        variables_name,
        vertices,
        center=None,
        axes=None,
        L=None,
        schedule=None,
        time_variant=None,
        how_to_update=None,
    ):

        box = cls(time_variant, how_to_update)
        
        axes = [""] if axes is None else axes
        if not isinstance(axes, list): #CHECK
            raise TypeError("The axes must be a list of string")
        box.naxes = len(axes)
        
        if center is None:
            center = np.array([0, 0])

        simplices, arrows, extremes = exbox_boundaries(vertices, center)

        for i, couple in enumerate(simplices):
            box.constraints.append(
                Constraint(variables_name[couple[0]], extremes[i], axes, arrows[i], center, L, schedule)
            )
            box.constraints.append(
                Constraint(variables_name[couple[1]], extremes[i], axes, arrows[i], center, L, schedule)
            )

        return box

    def apply_transforms(self, transforms: list[list[np.ndarray]]):
        for ic in range(len(transforms)):
            for il in range(len(transforms[ic])):
                new_extrem = transforms[ic][il][self.naxes, self.naxes] * self.constraints[ic].extreme[il]
                self.constraints[ic].update(extreme = new_extrem)

                transforms[ic][il][self.naxes, self.naxes] = 1

                new_arrow = np.ones(self.naxes + 1)
                new_arrow[:self.naxes] = self.constraints[ic].arrow[il, :]
                new_arrow = transforms[ic][il] @ new_arrow
                self.constraints[ic].update(arrow = new_arrow[:self.naxes])

                new_center = transforms[ic][il][:self.naxes, self.naxes] + self.constraints[ic].center[il, :]
                self.constraints[ic].update(center = new_center)

    def broadcast_uniform_transform(self, transform: np.ndarray):
        transforms = [[]] * len(self.constraints)
        # TODO: nlines can't be NONE or non-int  
        for ic in range(len(self.constraints)):
            for il in range(self.constraints[ic].nlines):
                transforms[ic].append(transform)
        return transforms

    def broadcast_nlines_transforms(self, transforms: list[np.ndarray]):
        return ([transforms] * len(self.constraints))

    def broadcast_uniform_translation(self, translation: np.ndarray, space: SPACE):
        if (space == SPACE.TS):
            transform = np.eye(self.naxes+1, self.naxes+1)
            transform[:self.naxes, self.naxes] = translation
            return self.broadcast_uniform_transform(transform)
        else:
            transforms = [[]] * len(self.constraints)
            for ic in range(len(self.constraints)):
                for il in range(self.constraints[ic].nlines):
                    transform = np.eye(self.naxes+1, self.naxes+1)
                    transform[:self.naxes, self.naxes] = self.constraints[ic].SS_to_TS(translation)
                    transforms[ic].append(transform)
            return transforms

    def broadcast_nlines_translations(self, translations: list[np.ndarray], space: SPACE):
        if (space == SPACE.TS):
            transforms = [[]] * len(translations)
            for i in range(len(translations)):
                transforms[i] = np.eye(self.naxes+1, self.naxes+1)
                transforms[i][:self.naxes, self.naxes] = translations[i]
            return self.broadcast_nlines_transforms(transforms)
        else:
            transforms = [[]] * len(self.constraints)
            for ic in range(len(self.constraints)):
                for il in range(self.constraints[ic].nlines):
                    transform = np.eye(self.naxes+1, self.naxes+1)
                    transform[:self.naxes, self.naxes] = self.constraints[ic].SS_to_TS(translations[il])
                    transforms[ic].append(transform)
            return transforms

    def broadcast_uniform_rotation(self, rotation: np.ndarray, space: SPACE):
        if (space == SPACE.TS):
            transform = np.eye(self.naxes+1, self.naxes+1)
            transform[:self.naxes, :self.naxes] = rotation
            return self.broadcast_uniform_transform(transform)
        else:
            print("Using SS rotations on a box is not implemented")
            return []

    def broadcast_nlines_rotations(self, rotations: list[np.ndarray], space: SPACE):
        if (space == SPACE.TS):
            transforms = [[]] * len(rotations)
            for i in range(len(rotations)):
                transform = np.eye(self.naxes+1, self.naxes+1)
                transform[:self.naxes, :self.naxes] = rotations[i]
                transforms[i] = transform
            return self.broadcast_nlines_transforms(transforms)
        else:
            print("Using SS rotations on a box is not implemented")
            return []

    def broadcast_uniform_rotation(self, rotation: np.ndarray, rotation_center: np.ndarray, space: SPACE):
        if (space == SPACE.TS):
            rotation_transform = np.eye(self.naxes+1, self.naxes+1)
            rotation_transform[:self.naxes, :self.naxes] = rotation

            first_translation_transform = np.eye(self.naxes+1, self.naxes+1)
            first_translation_transform[:self.naxes, self.naxes] = -rotation_center
            
            second_translation_transform = np.eye(self.naxes+1, self.naxes+1)
            second_translation_transform[:self.naxes, self.naxes] = rotation_center

            transform = second_translation_transform @ rotation_transform @ first_translation_transform

            return self.broadcast_uniform_transform(transform)
        else:
            print("Using SS rotations on a box is not implemented")
            return []

    def broadcast_nlines_rotations(self, rotations: list[np.ndarray], rotation_centers: list[np.ndarray], space: SPACE):
        if (space == SPACE.TS):
            transforms = [[]] * len(rotations)
            for i in range(len(rotations)):
                rotation_transform = np.eye(self.naxes+1, self.naxes+1)
                rotation_transform[:self.naxes, :self.naxes] = rotations[i]

                first_translation_transform = np.eye(self.naxes+1, self.naxes+1)
                first_translation_transform[:self.naxes, self.naxes] = -rotation_centers[i]
                
                second_translation_transform = np.eye(self.naxes+1, self.naxes+1)
                second_translation_transform[:self.naxes, self.naxes] = rotation_centers[i]

                transforms[i] = second_translation_transform @ rotation_transform @ first_translation_transform

            return self.broadcast_nlines_transforms(transforms)
        else:
            print("Using SS rotations on a box is not implemented")
            return []

    def broadcast_uniform_scale(self, scale: float):
        transform = np.eye(self.naxes+1, self.naxes+1)
        transform[self.naxes, self.naxes] = scale
        return self.broadcast_uniform_transform(transform)

    def broadcast_nlines_scales(self, scales: list[float]):
        transforms = [np.eye(self.naxes+1, self.naxes+1)] * len(scales)
        for i in range(len(scales)):
            transforms[i][self.naxes, self.naxes] = scales[i]
        return self.broadcast_nlines_transforms(transforms)
        
    def set_uniform_position(self, position: np.ndarray):
        for ic in range(len(self.constraints)):
            for il in range(len(self.constraints[ic].nlines)):
                self.constraints[ic].update(center = position)

    def set_nlines_position(self, positions: list[np.ndarray]):
        for ic in range(len(self.constraints)):
            for il in range(self.constraints[ic].nlines):
                self.constraints[ic].update(center = positions[il])

    def broadcast_uniform_safety_margin(self, margin: float):
        transforms = [[]] * len(self.constraints)
        for ic in range(len(self.constraints)):
            for il in range(len(self.constraints[ic].nlines)):
                scale = 1 + (self.constraints[ic].extrem / margin)
                transform = np.eye(self.naxes+1, self.naxes+1)
                transform[self.naxes, self.naxes] = scale
                transforms[ic].append(transform)
        return transforms

    def broadcast_nlines_safety_margin(self, margins: list[float]):
        transforms = [[]] * len(self.constraints)
        for ic in range(len(self.constraints)):
            for il in range(self.constraints[ic].nlines):
                scale = 1 + (self.constraints[ic].extrem / margins[il])
                transform = np.eye(self.naxes+1, self.naxes+1)
                transform[self.naxes, self.naxes] = scale
                transforms[ic].append(transform)
        return transforms

    def reschedule(self, new_schedule):
        self.schedule = new_schedule
        for boundary in self.constraints:
            boundary.update(schedule=self.schedule)

    def is_feasible(self, points, space="SS"):
        if not isinstance(points, list):
            points = [points]

        feasible = []
        for point in points:
            feasible.append(
                all([limit.is_feasible(point, space) for limit in self.constraints])
            )
        return feasible

    def update(self, **kargs):
        self.__figuring_out(self, **kargs)


def box_boundaries(vertices):
    """vertices is an ndarray with one vertex per row."""
    vertices = vertices.astype("float64")
    n, dim = vertices.shape
    center = np.sum(vertices, axis=0) / n

    if dim == 1:
        simplices = np.array([[0], [1]])
        arrows = np.ones([2, 1])

    elif dim == 2:
        simplices = sp.ConvexHull(vertices).simplices

        directions = vertices[simplices[:, 0]] - vertices[simplices[:, 1]]

        arrows = np.hstack(
            [directions[:, 1].reshape(-1, 1), -directions[:, 0].reshape(-1, 1)]
        )

    elif dim == 3:
        simplices = sp.ConvexHull(vertices).simplices

        d0 = vertices[simplices[:, 0]] - vertices[simplices[:, 2]]
        d1 = vertices[simplices[:, 1]] - vertices[simplices[:, 2]]

        arrows = np.hstack(
            [
                (d0[:, 1] * d1[:, 2] - d0[:, 2] * d1[:, 1]).reshape(-1, 1),
                (d0[:, 2] * d1[:, 0] - d0[:, 0] * d1[:, 2]).reshape(-1, 1),
                (d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]).reshape(-1, 1),
            ]
        )

    for i, arrow in enumerate(arrows):
        arrows[i] = arrow / np.linalg.norm(arrow)

    furthest_vertex = vertices[simplices[:, 0]]
    extremes = np.sum(arrows * (furthest_vertex - center), axis=1).reshape([-1, 1])

    return arrows, extremes, center

def exbox_boundaries(vertices, center):
    """vertices is an ndarray with one vertex per row."""
    vertices = vertices.astype("float64")
    n, dim = vertices.shape

    # if dim == 1:
    #     simplices = np.array([[0], [1]])
    #     arrows = np.ones([2, 1])

    if dim == 2:
        simplices = sp.ConvexHull(vertices).simplices

        directions = vertices[simplices[:, 0]] - vertices[simplices[:, 1]]

        arrows = np.hstack(
            [directions[:, 1].reshape(-1, 1), -directions[:, 0].reshape(-1, 1)]
        )

    # elif dim == 3:
    #     simplices = sp.ConvexHull(vertices).simplices

    #     d0 = vertices[simplices[:, 0]] - vertices[simplices[:, 2]]
    #     d1 = vertices[simplices[:, 1]] - vertices[simplices[:, 2]]

    #     arrows = np.hstack(
    #         [
    #             (d0[:, 1] * d1[:, 2] - d0[:, 2] * d1[:, 1]).reshape(-1, 1),
    #             (d0[:, 2] * d1[:, 0] - d0[:, 0] * d1[:, 2]).reshape(-1, 1),
    #             (d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]).reshape(-1, 1),
    #         ]
    #     )

    for i, arrow in enumerate(arrows):
        arrows[i] = arrow / np.linalg.norm(arrow)

    furthest_vertex = -1e-8*vertices[simplices[:, 0]]
    extremes = np.sum(arrows * (furthest_vertex - center), axis=1).reshape([-1, 1])

    return simplices, arrows, extremes


