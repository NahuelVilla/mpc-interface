#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:04:15 2021

@author: nvilla
"""
import numpy as np
import scipy.spatial as sp
import mpc_core.tools as use

## TODO: make some visualization of the constraints graphically

class Constraint:
    def __init__(self, variable, extreme, axes=None, arrow=None,
                 center=None, L=None, schedule=None):
        """ 
        ARGUMENTS:
            
            variable : dict, {name: String, combination: dict }
            
            axes : list of strings ex: ["_x", "_y", "_z"] always starting "_"
            
            arrow :   ndarray with shape [m, len(axes)] or list
            
            extreme : ndarray with shape [m, 1] or single number, or list.
            
            center :  ndarray with shape [m, len(axes)] or list
            
            L : list of [len(axes)] ndarrays each one with shape [m, t]
            
            schedule : range with t elements with t <= horizon_lenght
        
        The null value of L is an empty list [] and the null value of 
        schedule is range(0). These values can be used in the update 
        function to remove them.
        
        Each instance of this class provides instructions to generate 
        m (or t (or horizon_lenght)) lineal constraints of the form:
            
            arrow * ( V - center ) < extreme
            
        where V is:
            
            V = [  Lx @ v_x[schedule], Ly @ v_y[schedule], ... ]
            
        based on the variable v = [v_x, v_y] which is defined by 'variable'.
        """
        self.variable = variable
        self.axes = [""] if axes is None else axes
        if not isinstance(self.axes, list):
            raise TypeError("The axes must be a list of string")
        self.axes_len = len(self.axes)    
        
        self.schedule = range(0) if schedule is None else schedule
        if L is None:
            self.L = []
        else:
            self.arrange_L(L)
        
        self.extreme = np.array(extreme).reshape([-1, 1]).astype(float)
        self.initialize_geometry(arrow, center)
        self.check_geometry()
        self.normalize()
    
    @classmethod   
    def copy(cls, limit):
        constraint = cls(limit.variable, 
                         limit.extreme.copy(), 
                         limit.axes, 
                         limit.arrow.copy(), 
                         limit.center.copy(),
                         limit.L,
                         limit.schedule)
        return constraint
        
    def arrange_L(self, L):
        """ This function requires an updated schedule."""
        self.L = L if isinstance(L, list) else [L]
        
        if len(self.L) == 1:
            self.L = self.L*self.axes_len
        elif len(self.L) not in (self.axes_len, 0):
            raise IndexError("'L' must have 0, 1 or len(axes) = {} "+
                             "elements".format(self.axes_len))
            
        if self.schedule and np.any([l.shape[-1] != self.t 
                                     for l in self.L]):
            raise ValueError("arrays in L must have {} ".format(self.t)+
                             "columns, which is given by the 'schedule'.")
    
    def initialize_geometry(self, arrow, center):
        if arrow is None:    
            if self.axes_len == 1: 
                self.arrow = np.ones(self.extreme.shape)
                
            else:
                raise ValueError("When using multiple axes, "+
                "some normal direction 'arrow' must be provided")
        
        else: 
            self.arrow = np.array(arrow).reshape([-1, self.axes_len])       
        
        if center is None:
            self.center = np.zeros([1, self.axes_len]) 
                     
        else: 
            self.center = np.array(center).reshape([-1, self.axes_len])   
            
    def check_geometry(self):     
        rows = np.array([self.arrow.shape[0], 
                         self.center.shape[0],
                         self.extreme.shape[0]])
        rows_not_1 = rows[rows != 1]
        
        if np.any(rows_not_1) and np.any(rows_not_1 != rows_not_1[0]):
            raise ValueError("The number of rows in 'arrow', 'center' and "+
                             "'extreme' must be equal or 1, but they are "+
                             "{} respectively".format(rows))
        if self.axes_len > 1:
            cols = np.array([self.arrow.shape[1], self.center.shape[1]])
            if np.any(cols != self.axes_len):
                raise IndexError(("'arrow' and 'center' have {} columns "+
                                  "but they must have {}, one per "+
                                  "axis.").format(cols, self.axes_len)) 
    
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
        
        sizes = np.array([self.arrow.shape[0], 
                          self.center.shape[0],
                          self.extreme.shape[0]])
        if np.all(sizes==1):
            return None
        
        sizes_not_1 = sizes[sizes != 1]
        return sizes_not_1[0]
    
#    def forecast(self, N=None):
#        
#        
#        if N:
#            eN = self.extreme.shape[0]
#            
#            self.extreme = np.pad(self.extreme, [[0,0],[N,0]], mode="edge")
#        
#        
#        
        
        
    def broadcast(self, N=None):
        if self.m or self.t:
            self.extreme = np.resize(self.extreme, [self.nlines, 1])
            self.arrow = np.resize(self.arrow, [self.nlines, self.axes_len])
            self.center = np.resize(self.center, [self.nlines, self.axes_len])
            
            if N and N != self.nlines:
                raise ValueError("This bound can only have 1 or "+
                                 "{} elements.".format(self.m 
                                                        if self.m else self.t))
            return
        
        if N:
            self.extreme = np.resize(self.extreme, [N, 1])
            self.arrow = np.resize(self.arrow, [N, self.axes_len])
            self.center = np.resize(self.center, [N, self.axes_len])
            return
        
    def matrices(self):
        if self.L:
            return [self.arrow[:,i][:,None]*l for i, l in enumerate(self.L)]
        else:
            return [self.arrow[:,i][:,None] for i in range(self.axes_len)]
            
    def normalize(self):
        if self.arrow.shape[0] != self.extreme.shape[0]:
            if self.arrow.shape[0] == 1:
                self.arrow = np.resize(self.arrow, 
                                       [self.extreme.shape[0], self.axes_len])
            elif self.extreme.shape[0] == 1:
                self.extreme = np.resize(self.extreme, 
                                         [self.arrow.shape[0], 1])
            
        for i, extreme in enumerate(self.extreme):
            # We adapt the arrow to have always positive extreme
            if extreme < 0:
                self.extreme[i] = -self.extreme[i]
                self.arrow[i] = -self.arrow[i]
#            Unit vectors in arrow?

    def bound(self):
        return self.extreme + np.sum(self.arrow * 
                                     self.center, axis=1).reshape([-1, 1])
                        
    def update(self, extreme=None, arrow=None, center=None,
               L=None, schedule=None):
        if schedule is not None:
            self.schedule = schedule
            
        if L is not None:
            self.arrange_L(L)
        
        if extreme is not None:
            self.extreme = np.array(extreme).reshape([-1, 1])
        if arrow is not None:
            self.arrow = np.array(arrow).reshape([-1, self.axes_len])
        if center is not None:
            self.center = np.array(center).reshape([-1, self.axes_len]) 
        
        if extreme is not None or arrow is not None:
            self.check_geometry()
            self.normalize()
        elif center is not None:
            
            self.check_geometry()
    
    def is_feasible(self, points, space="SS"):
        """ Introduce a list of points to verify the feasibility of each
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
        """ The ss_point must be arranged with one column per TS_axis.
        ex: ss_point = [p_x, p_y, p_z] = [[pv_x, pv_y, pv_z],
                                          [pw_x, pw_y, pw_z],]
        where x, y and z are TS axes, and v and w are SS axes
        """
        if self.L:
            return np.vstack([
                    (l@p.T) for l, p in zip(self.L, np.transpose(ss_point))
                    ]).T
        return ss_point
        
    def _is_feasible_SS(self, ss_point):
        ts_point = self.SS_to_TS(ss_point)
        return self._is_feasible_TS(ts_point)
    
    def _is_feasible_TS(self, ts_point):
        return np.sum(self.arrow*(ts_point - self.center), axis=1) < self.extreme
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        if self.axes != [""]:
            axes = "_"+"".join(axis[1:] for axis in self.axes)
        else:
            axes = ""
        
        text = "\nvariable:\n"+" "*7+'"'+self.variable+axes+'"' 
        text+= "\nwith L = " +str(",\n".join(str(l) for l in self.L)) if self.L != [] else ""
        text+= "\narrow:\n"+" "*7+ ("\n"+" "*7).join(str(arrow) for arrow in self.arrow)
        text+= "\ncenter:\n"+" "*7+ ("\n"+" "*7).join(str(center) for center in self.center)
        text+= "\nextreme:\n"+" "*7+ ("\n"+" "*7).join(str(extreme) for extreme in self.extreme)
        text+= "\n"
        return text
    
class Boxel:
    def __init__(self, time_variant=None, how_to_update=None):
        
        self.constraints = []
        self.vertices = []
        self.axes = [""]
        self.axes_len = 1
        
        self.center_TS = []
        self.center_SS = []
        self.orientation = []
        
        self.schedule = range(0)
        
        self.scale = [1]
        self.safety_margins = [0]
        
        self.dimention_SS = 0
        self.dimention_TS = 0
        
        self.cast = 1
        
        if how_to_update is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update
            
    @classmethod        
    def task_space(cls, variable, vertices, axes=None, schedule=None,
                   time_variant=None, how_to_update=None):
        
        box = cls(time_variant, how_to_update)
        
        arrows, extremes, center = box_boundaries(vertices)
        for i, extreme in enumerate(extremes):
            box.constraints.append(Constraint(variable, extreme, axes,
                                              arrows[i], center, None, schedule))           
        
        box.variable = variable
        box.vertices = vertices
        
        box.ss_dimention = box.ts_dimention = vertices.shape[1]
        
        box.axes = box.constraints[0].axes
        box.axes_len = len(box.axes)
        
        box.schedule = schedule
        box.center_TS = center
        box.center_SS = center
        box.orientation = [np.eye(box.axes_len)]
        box.scale = [1]
        box.safety_margins = [0]
        
        box.cast = 1
        return box
    
    @classmethod        
    def state_space(cls, variable, vertices, axes=None, schedule=None,
                   time_variant=None, how_to_update=None):
        
        box = cls(time_variant, how_to_update)
        
        arrows_SS, extremes_SS, center_SS = box_boundaries(vertices)
        center = np.sum(arrows_SS*center_SS, axis=1).reshape([-1, 1])
        
        for i, extreme in enumerate(extremes_SS):
            box.constraints.append(Constraint(variable, extreme, axes=axes,
                                    center=center[i], L=arrows_SS[i],
                                    schedule=schedule))
        
        box.variable = variable
        box.vertices = vertices
        
        box.ss_dimention = vertices.shape[1]
        box.ts_dimention = 1
        
        box.axes = box.constraints[0].axes
        box.axes_len = len(box.axes)
        
        box.schedule = schedule
        box.center_SS = center_SS.reshape([-1, 1])
        box.center_TS = 0
        box.orientation = [np.eye(box.axes_len)]
        box.scale = [1]
        box.safety_margins = [0]
        
        box.cast = 1
        return box
    
#    def forecast(self, new_cast):
#        for bound in self.constraints:
#            bound.broadcast(new_cast)
#        
#        
#        
        
               
class Box:
    def __init__(self, time_variant=None, how_to_update=None):
        
        self.constraints = []
        self.o_constraints = []
        
        self.ts_vertices = np.array([])
        self.ss_vertices = np.array([])
        self.ts_center = np.array([])
        self.ss_center = np.array([])
        
        self.schedule = range(0)
        self.safety_margin = 0
        self.axes = [""]
        self.ss_dimention = 0
        self.ts_dimention = 0
        
        self.time_variant = time_variant
        
        if how_to_update is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update
    
    @property
    def m(self):
        return self.constraints[0].m
    
    @property
    def t(self):
        return self.constraints[0].t
    
    @classmethod        
    def task_space(cls, variable, vertices, axes=None, L=None, schedule=None,
                   time_variant=None, how_to_update=None):
        
        box = cls(time_variant, how_to_update)
        
        arrows, extremes, center = box_boundaries(vertices)
        for i, extreme in enumerate(extremes):
            box.constraints.append(Constraint(variable, extreme, axes,
                                              arrows[i], center, L, schedule))
            box.o_constraints.append(Constraint.copy(box.constraints[-1]))
        
        box.ss_dimention = box.ts_dimention = vertices.shape[1]
        box.axes = box.constraints[0].axes
        box.ts_center = center
        box.ss_center = np.zeros(center.shape)
        box.ts_vertices = vertices
        box.schedule = schedule
        return box
    
    @classmethod        
    def state_space(cls, variable, vertices, axes=None, schedule=None,
                   time_variant=None, how_to_update=None):
        
        box = cls(time_variant, how_to_update)
        
        arrows_SS, extremes_SS, center_SS = box_boundaries(vertices)
        center = np.sum(arrows_SS*center_SS, axis=1).reshape([-1, 1])
        
        for i, extreme in enumerate(extremes_SS):
            box.constraints.append(Constraint(variable, extreme, axes=axes,
                                    center=center[i], L=arrows_SS[i],
                                    schedule=schedule))
            box.o_constraints.append(Constraint.copy(box.constraints[-1]))
        
        box.ss_dimention = vertices.shape[1]
        box.ts_dimention = 1
        box.axes = box.constraints[0].axes
        box.ss_center = center_SS.reshape([-1, 1])
        box.ts_center = 0
        box.ss_vertices = vertices
        box.schedule = schedule
        return box
    
    ## ~~RELOCATIONS~~ ## 

    def _get_dimention(self, size):
        m = self.m
        t = self.t
        bound_size = m if m else (t if t else 1)
        
        if bound_size != size:
            if bound_size == 1:
                for bound in self.o_constraints:
                    bound.broadcast(size)
                return size
            if size == 1:
                return bound_size
            else:
                raise IndexError("The new locations can be 1 or"+
                                 "{}, but {} were introduced".format(bound_size,
                                                                     size))
        return size
        
    def recenter_in_TS(self, new_center):
        """ Danger: If the box defines a set in the SS, this function would
        change the shape and size of such set. This deformation can be 
        corrected by executing 'box.recenter_in_TS(original_center)'.
        For a safer recentering, use 'recenter_in_SS'.
        """
        self.ts_center = np.array(new_center) 
        for boundary in self.constraints:
            boundary.update(center=self.ts_center)
    
    def recenter_in_SS(self, new_center):
        c_shape = np.shape(new_center)
        correct_forms = ((self.ss_dimention, self.ts_dimention), 
                         (self.ts_dimention,))
        if not c_shape in correct_forms:
            raise ValueError(("The 'new_center' must have {} "+
                              "rows and {} columns, but its shape "+
                              "is {}").format(self.ss_dimention,
                                              self.ts_dimention,
                                              np.shape(new_center)))
        self.ss_center = np.array(new_center)
        for boundary in self.constraints:
            center = boundary.SS_to_TS(new_center)
            boundary.update(center=center)
    
    def reschedule(self, new_schedule):
        self.schedule = new_schedule
        for boundary in self.constraints:
            boundary.update(schedule=self.schedule)
            
    def reorient_in_TS(self, orientations):
        if not isinstance(orientations, list):
            orientations = [orientations]
        orientations *= self._get_dimention(len(orientations))
        
        for i, boundary in enumerate(self.constraints):
            arrows = self.o_constraints[i].arrow
            
            new_arrows = [n @ R.T for R, n in zip(orientations, arrows)]
            boundary.update(arrow=np.vstack(new_arrows))
            
    def rescale(self, factors):
        if not isinstance(factors, np.ndarray):
            factors = np.array(factors).flatten()
        factors = np.resize(factors, self._get_dimention(factors.size))
        
        for i, boundary in enumerate(self.constraints):
            extremes = self.o_constraints[i].extreme
            
            boundary.update(extreme = extremes * factors)
        
        if self.safety_margin > 0:
            margin = self.safety_margin; self.safety_margin = 0
            self.set_safety_margin(margin)
            
    def set_safety_margin(self, margins):
        if not isinstance(margins, np.ndarray):
            margins = np.array(margins).flatten()
        margins = np.resize(margins, self._get_dimention(margins.size))
        
        for boundary in self.constraints:
            arrow_norms = np.linalg.norm(boundary.arrow, axis=1)
            
            new_extreme = boundary.extreme - arrow_norms * (margins -
                                                            self.safety_margin)
            boundary.update(extreme = new_extreme)
        self.safety_margin = margins
       
            
    ## ~~TRANSFORMATIONS~~ ## 
    ## TODO: make all the incremental transformations based on the relocations.
    def translate_in_TS(self, translation):
        self.ts_center += translation
        for boundary in self.constraints:
            new_center = boundary.center + translation
            boundary.update(center=new_center)
    
    def translate_in_SS(self, translation):
        c_shape = np.shape(translation)
        correct_forms = ((self.ss_dimention, self.ts_dimention), 
                         (self.ts_dimention,))
        if not c_shape in correct_forms:
            raise ValueError(("The 'new_center' must have {} "+
                              "rows and {} columns, but its shape "+
                              "is {}").format(self.ss_dimention,
                                              self.ts_dimention,
                                              np.shape(translation)))
        self.ss_center += np.array(translation)
        for boundary in self.constraints:
            center = boundary.center + boundary.SS_to_TS(translation)
            boundary.update(center=center)
    
    def rotate_in_TS(self, rotations):
        if not isinstance(rotations, list):
            rotations = [rotations]
            
        for limit in self.constraints:
            limit.broadcast(len(rotations))
            
        N = self.constraints[0].arrow.shape[0]
            
        if len(rotations) != N:
            if len(rotations) == 1:
                rotations *= N
            else:
                raise IndexError("'rotations' must contain 1 or "+
                              "{} rotation matrices".format(N))
        
        for boundary in self.constraints:
            arrows = boundary.arrow
            
            new_arrows = [n @ R.T for R, n in zip(rotations, arrows)]
            boundary.update(arrow=np.vstack(new_arrows))
            
    def rotate_in_SS(self, rotations):
        raise NotImplementedError("Maybe later.") 
        
    ## TODO: Make visualization of the box using arrow, extreme and center.
            
    def is_feasible(self, points, space = "SS"):
        if not isinstance(points, list):
            points = [points]
        
        feasible = []
        for point in points:
            feasible.append(
                    all([limit.is_feasible(point, space) 
                         for limit in self.constraints])
                           )
        return feasible
    
    def update(self, **kargs): 
        self.__figuring_out(self, **kargs)
        
def box_boundaries(vertices):
    """ vertices is an ndarray with one vertex per row."""
    vertices = vertices.astype("float64")
    n, dim = vertices.shape
    center = np.sum(vertices, axis=0)/n
    
    if dim == 1:
        simplices = np.array([[0], [1]])
        arrows = np.ones([2, 1])
    
    elif dim == 2:
        simplices = sp.ConvexHull(vertices).simplices
        
        directions = (vertices[simplices[:, 0]] - 
                      vertices[simplices[:, 1]])
        
        arrows = np.hstack([ directions[:, 1].reshape(-1, 1),
                            -directions[:, 0].reshape(-1, 1)])
                
    elif dim == 3:
        simplices = sp.ConvexHull(vertices).simplices
        
        d0 = (vertices[simplices[:, 0]] - 
              vertices[simplices[:, 2]])
        d1 = (vertices[simplices[:, 1]] - 
              vertices[simplices[:, 2]])
        
        arrows = np.hstack([(d0[:, 1]*d1[:, 2]-
                             d0[:, 2]*d1[:, 1]).reshape(-1, 1), 
                            (d0[:, 2]*d1[:, 0]-
                             d0[:, 0]*d1[:, 2]).reshape(-1, 1),
                            (d0[:, 0]*d1[:, 1]-
                             d0[:, 1]*d1[:, 0]).reshape(-1, 1)])

    for i, arrow in enumerate(arrows):
        arrows[i] = arrow/np.linalg.norm(arrow)
        
    furthest_vertex = vertices[simplices[:, 0]]
    extremes = np.sum(arrows * (furthest_vertex - center), 
                      axis=1).reshape([-1, 1])
                              
    return arrows, extremes, center 

    
    
    