# %%
# Formulation of foot balancing a broom
from mpc_interface.dynamics import ExtendedSystem, ControlSystem
from mpc_interface.goal import Cost
from mpc_interface.body import Formulation
from mpc_interface.combinations import LineCombo
from mpc_interface.restrictions import Constraint, Box
import mpc_interface.tools as now
import numpy as np

## CONFIG
broom_length =  1.37 # [m]
foot_height = 0.10 # [m]
CoM_height = foot_height + (0.60 * broom_length)
gravity = [0, 0, -9.81]
w = np.sqrt(-gravity[2] / CoM_height)

## LIP
input_names = ["dddc"]
state_names = ["c", "dc", "ddc"]
axes = ["_x", "_y"]
horizon_length = 3
LIP = ControlSystem.from_name("J->CCC", input_names, state_names, axes, tau=0.1)
LIP_ext = ExtendedSystem.from_control_system(LIP, "s", horizon_length)

## DEFINITIONS
some_defs = {}
for axis in axes:
    p = LineCombo({"c" + axis: 1, "ddc" + axis: -1/w**2})
    d = LineCombo({"c" + axis: 1, "p" + axis: -1})

    some_defs.update(
        {
            "p" + axis: p,
            "d" + axis: d
        }
    )

## CONSTRAINT
vertices = []
r = 0.10 # [m]
n = 16
for i in range(n - 1):
    portion = float(i) / float(n)
    vertices.append(
        [r * np.cos(2 * np.pi * portion),
        r * np.sin(2 * np.pi * portion)]
    )
vertices = np.array(vertices)
bounding_box = Box.task_space(
    variable="d",
    vertices=vertices,
    axes=axes,
)

# #COSTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# zero_velocity_x = Cost("dc", 1, axes=["_x"])
# zero_velocity_y = Cost("dc", 1, axes=["_y"])
zero_acceleration_x = Cost("ddc", 1, axes=["_x"])
zero_acceleration_y = Cost("ddc", 1, axes=["_y"])
minimum_jerk = Cost("dddc", 0.001, axes=axes)

## Formulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
form = Formulation()
form.incorporate_dynamics("broom", LIP_ext)
form.incorporate_definitions(some_defs)
form.incorporate_box("bounding_box", bounding_box)
# form.incorporate_goal("zero_velocity_x", zero_velocity_x)
# form.incorporate_goal("zero_velocity_y", zero_velocity_y)
form.incorporate_goal("zero_acceleration_x", zero_acceleration_x)
form.incorporate_goal("zero_acceleration_y", zero_acceleration_y)
form.incorporate_goal("minimum_jerk", minimum_jerk)

form.identify_qp_domain(["dddc_x", "dddc_y"])
form.make_preview_matrices()

# Define the given values
given_values = {
    "s0_x": np.array([[0.0], [10.0], [0.3]]),
    "s0_y": np.array([[0.0], [1.0], [0.2]])
}

A, h, Q, q = form.generate_all_qp_matrices(form.arrange_given(given_values))
Q = Q / 2 + Q.T / 2 # make matrix symmetric

# solve a qp

from matplotlib import pyplot as plt
from qpsolvers import osqp_solve_qp
import scipy.sparse as sy

Q = sy.csc_matrix(Q)
A = sy.csc_matrix(A)

X = osqp_solve_qp(P=Q, q=q, A=A, h=h).reshape([-1, 1])

time = np.linspace(0, 2, horizon_length)
c_x = form.preview(form.arrange_given(given_values), X, "c_x")
c_y = form.preview(form.arrange_given(given_values), X, "c_y")
dc_x = form.preview(form.arrange_given(given_values), X, "dc_x")
dc_y = form.preview(form.arrange_given(given_values), X, "dc_y")
ddc_x = form.preview(form.arrange_given(given_values), X, "ddc_x")
ddc_y = form.preview(form.arrange_given(given_values), X, "ddc_y")
dddc_x = form.preview(form.arrange_given(given_values), X, "dddc_x")
dddc_y = form.preview(form.arrange_given(given_values), X, "dddc_y")

# plt.plot(time, c_x, label="c_x")
# plt.plot(time, c_y, label="c_x")
plt.plot(time, dc_x, label="dc_x")
plt.plot(time, dc_y, label="dc_y")
# plt.plot(time, ddc_x, label="ddc_x")
# plt.plot(time, ddc_y, label="ddc_y")
plt.plot(time, dddc_x, label="dddc_x")
plt.plot(time, dddc_y, label="dddc_y")
leg = plt.legend(loc='best')

# plt.grid()
# plt.plot(c_x, c_y)

# %%
