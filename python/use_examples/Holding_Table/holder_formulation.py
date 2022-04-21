#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:15:40 2022

@author: nvilla
"""

import numpy as np

from mpc_core.dynamics import ControlSystem, ExtendedSystem, DomainVariable
from mpc_core.body import Formulation
import mpc_core.tools as now
from mpc_core.restrictions import Box
from mpc_core.goal import Cost
from mpc_core.combinations import LineCombo

import holder_configuration as config


def formulate_biped(conf):

    w = conf.omega
    horizon_lenght = conf.horizon_lenght
    step_samples = conf.step_samples
    system = conf.system
    #### DYNAMICS AND DOMAIN VARIABLES ~~~~~~~~~~~~~~~~~~~~
    axes = ["_x", "_y"]
    ##~Steps~##
    E = now.plan_steps(horizon_lenght, 0, regular_time=step_samples)
    F = np.ones([horizon_lenght, 1])

    steps = ExtendedSystem(
        "Ds",
        "s",
        "s",
        S=F,
        U=E,
        axes=axes,
        how_to_update_matrices=now.update_step_matrices,
        time_variant=True,
    )
    steps.define_output(
        "stamps", {"s0": 1, "Ds": 1}, time_variant=True, how_to_update=now.adapt_size
    )

    ##~LIP~##
    LIP = ControlSystem.from_name(system, axes, tau=conf.mpc_period, omega=w)
    LIP_ext = ExtendedSystem.from_cotrol_system(LIP, "x", horizon_lenght)

    ##~Non-Linearity~##
    bias = DomainVariable("n", horizon_lenght, axes)

    ##~Feet Orientations~##
    n_steps = E.shape[1]
    orientation = DomainVariable(
        "yawl", n_steps, time_variant=True, how_to_update_size=now.count_yawls
    )

    ## EXTRA DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_coeff = np.diag(np.ones([horizon_lenght - 1]), 1)

    some_defs = {}
    if system == "J->CCC":
        for axis in axes:
            LIP_B = LineCombo({"CoM" + axis: 1, "CoM_ddot" + axis: -1 / w**2})
            BpNmS = LineCombo({"b" + axis: 1, "n" + axis: n_coeff, "s" + axis: -1})
            CmS = LineCombo({"CoM" + axis: 1, "s" + axis: -1})
            DCM = LineCombo({"CoM" + axis: 1, "CoM_dot" + axis: 1 / w})
            DCMmS = LineCombo({"DCM" + axis: 1, "s" + axis: -1})

            some_defs.update(
                {
                    "b" + axis: LIP_B,
                    "(b+n-s)" + axis: BpNmS,
                    "(c-s)" + axis: CmS,
                    "DCM" + axis: DCM,
                    "(DCM-s)" + axis: DCMmS,
                }
            )
        Vx = np.vstack(
            [np.eye(horizon_lenght), np.zeros([horizon_lenght, horizon_lenght])]
        )
        Vy = np.vstack(
            [np.zeros([horizon_lenght, horizon_lenght]), np.eye(horizon_lenght)]
        )
        C_xy = LineCombo({"CoM_x": Vx, "CoM_y": Vy})
        some_defs.update({"CoM_xy": C_xy})

    ## CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    support_vertices = now.make_simetric_vertices(conf.foot_corner)

    support_polygon = Box.task_space("(b+n-s)", support_vertices, axes)

    stepping_vertices = now.make_simetric_vertices(conf.stepping_corner)
    stepping_area = Box.task_space(
        "Ds",
        stepping_vertices,
        axes,
        how_to_update=now.update_stepping_area,
        time_variant=True,
    )
    stepping_area.update(
        step_count=0, n_next_steps=steps.domain["Ds_x"], xy_lenght=conf.stepping_center
    )

    terminal_constraint = Box.task_space(
        "(DCM-s)",
        support_vertices,
        axes,
        schedule=range(horizon_lenght - 1, horizon_lenght),
    )

    cop_safety_margin = conf.cop_safety_margin

    support_polygon.set_safety_margin(cop_safety_margin)
    terminal_constraint.set_safety_margin(cop_safety_margin)

    #    max_yawl = Constraint("yawl", 0.05)## TODO: implement this with 1D box
    #    min_yawl = Constraint("yawl", -0.05)
    yawl_limits = Box.task_space("yawl", np.array([[0.06], [-0.06]]))

    ## COSTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    relax_ankles_x = Cost(
        "(b+n-s)", conf.cost_weights["relax ankles"], aim=[0], axes=["_x"]
    )
    relax_ankles_y = Cost(
        "(b+n-s)", conf.cost_weights["relax ankles"], aim=[0], axes=["_y"]
    )
    minimum_jerk = Cost(
        "CoM_dddot", conf.cost_weights["minimize jerk"], aim=[0, 0], axes=axes
    )
    track_vel_x = Cost(
        "CoM_dot",
        conf.cost_weights["track velocity"],
        aim=conf.target_vel[0],
        axes=["_x"],
    )
    track_vel_y = Cost(
        "CoM_dot",
        conf.cost_weights["track velocity"],
        aim=conf.target_vel[1],
        axes=["_y"],
    )
    terminal_cost = Cost(
        "(DCM-s)",
        conf.cost_weights["terminal"],
        aim=[0, 0],
        axes=axes,
        schedule=range(horizon_lenght - 1, horizon_lenght),
    )
    #    orient_feet = Cost("yawl", 0.1, cross="CoM_xy")
    orient_feet = Cost("yawl", 0.1, aim=0)

    ## TODO: The cost for orientation is time varing. I have to make the funtions to
    ## compute and update the matrices of sin and cos as well as the matrices
    ## for average velociy.

    ##Formulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    form = Formulation()
    form.incorporate_dynamics("steps", steps)
    form.incorporate_dynamics("LIP", LIP_ext)
    form.incorporate_dynamics("bias", bias)
    form.incorporate_dynamics("orientation", orientation)
    form.incorporate_definitions(some_defs)
    form.incorporate_goal("relax ankles x", relax_ankles_x)
    form.incorporate_goal("relax ankles y", relax_ankles_y)
    form.incorporate_goal("minimize jerk", minimum_jerk)
    form.incorporate_goal("track vel_x", track_vel_x)
    form.incorporate_goal("track vel_y", track_vel_y)
    form.incorporate_goal("terminal_cost", terminal_cost)
    form.incorporate_goal("orient_feet", orient_feet)
    #    form.incorporate_constraint("max_yawl", max_yawl)
    #    form.incorporate_constraint("min_yawl", min_yawl)
    form.incorporate_box("yawl_limits", yawl_limits)
    form.incorporate_box("stepping area", stepping_area)
    form.incorporate_box("support_polygon", support_polygon)
    form.incorporate_box("terminal_Constraint", terminal_constraint)

    form.identify_qp_domain(["CoM_dddot_x", "Ds_x", "CoM_dddot_y", "Ds_y", "yawl"])  #
    form.make_preview_matrices()

    #    def update_orient_cost(cost, **kargs):
    #
    #        count = kargs["count"]
    #        N = kargs["N"]
    #        step_times = kargs["step_times"]
    #        old_yawls = kargs["old_yawls"]
    #
    #        m = now.n_predicted_steps(count, N, step_times)
    #        ## TODO: take the appropriate old_yawls and repeat the last one if needed.
    #        yL, y_ref =  now.linear_Rotations(old_yawls)
    #
    #        vL = now.step_average_velocity(count, N, step_times)
    #        cost.update(L=vL, cross_L=yL, cross_aim = y_ref)

    def update_this_formulation(body, **kargs):
        """
        The current implementation requires the following arguments to update
        the formulation:

            ### for the dynamic of stepes: ###
                Arguments:
                    count: current iteration number (count of mpc periods)

                Parameters:
                    horizon_lenght: number of samples of the horizon.
                    step_times: Times to land future steps.

            ### for the stepping constraint: ###
                Arguments:
                    step_count: current count of steps.

                Parameters:
                    n_next_steps: number of previewed steps
                    stepping_center : array from each step place to the center
                                      of next stepping area.
        """
        step_dynamics_keys = dict(step_times=kargs["step_times"], N=horizon_lenght)
        body.dynamics["steps"].update(**step_dynamics_keys)
        body.dynamics["orientation"].update(**step_dynamics_keys)

        stepping_constraint_keys = dict(
            step_count=kargs["step_count"],
            n_next_steps=body.dynamics["steps"].domain["Ds_x"],
            xy_lenght=conf.stepping_center,
        )
        body.constraint_boxes["stepping area"].update(**stepping_constraint_keys)

        body.constraint_boxes["support_polygon"].rotate_in_TS(now.rotation2D(0))
        body.constraint_boxes["terminal_Constraint"].rotate_in_TS(now.rotation2D(0))


        ## TODO: update the orient_feet cost.

        ## TODO: update the suport polygon constraint.
        ## TODO: update the terminal constraint.

    form.set_updating_rule(update_this_formulation)

    return form


if __name__ == "__main__":

    formulation = formulate_biped(config)
