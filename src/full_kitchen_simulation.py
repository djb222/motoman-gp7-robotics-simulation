# GP7_environment.py
# Build-only module (no env.launch, no env.hold)
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialgeometry import Cylinder
from ir_support import CylindricalDHRobotPlot
from math import pi

deg = pi/180

def build_GP7_environment(env, base=SE3(), name="GP7"):
    """
    Adds a Motoman GP7-with-stick to an existing Swift env and returns handles.
    Returns: robot, cyl_plot, stick_vis
    """
    # --- GP7 standard DH parameters ---
    d1 = 0.333
    a2 = 0.445
    a3 = 0.040
    d4 = 0.440
    d6 = 0.080

    # Manufacturer-like joint limits (rad)
    qlim = np.array([
    [-170*deg,  +170*deg],   # J1
    [ -65*deg,  +145*deg],   # J2
    [-150*deg,  +265*deg],   # J3
    [-190*deg,  +190*deg],   # J4
    [-125*deg,  +230*deg],   # J5
    [-360*deg,  +360*deg],   # J6
    ])

    links = [
        rtb.RevoluteDH(d=d1, a=0.0,  alpha=-pi/2, qlim = qlim[0]),   # J1
        rtb.RevoluteDH(d=0.0, a=a2,  alpha= 0.0, qlim=qlim[1]),    # J2
        rtb.RevoluteDH(d=0.0, a=a3,  alpha=-pi/2, qlim=qlim[2]),   # J3
        rtb.RevoluteDH(d=d4, a=0.0,  alpha= pi/2, qlim=qlim[3]),   # J4
        rtb.RevoluteDH(d=0.0, a=0.0, alpha=-pi/2, qlim=qlim[4]),   # J5
        rtb.RevoluteDH(d=d6, a=0.0,  alpha= 0.0, qlim=qlim[5]),    # J6
    ]

    
    
    robot = rtb.DHRobot(links, name=f"Motoman {name} (std DH)")
    robot.base = base
    robot.q = np.array([0, -90*deg, +90*deg, 0, 0, 0], dtype=float)

    # Cylindrical visual links (same style used elsewhere)
    cyl = CylindricalDHRobotPlot(robot, cylinder_radius=0.04, multicolor=True)
    gp7_vis = cyl.create_cylinders()
    env.add(gp7_vis)

    # Orange “stick” as a loose object (pick-and-place)
    STICK_LEN = 0.18
    STICK_RAD = 0.012
    SPOON_LOCAL = np.array([0.48, -0.12, 0.22])   # in GP7 base-frame coords

# Place the stick in world using the GP7 base pose
    stick = Cylinder(radius=STICK_RAD, length=STICK_LEN,
                    pose=base * SE3(*SPOON_LOCAL),
                    color=[0.95, 0.6, 0.2, 1.0])
    env.add(stick)

# Keep TCP tool frame oriented "down" for later stirring
    robot.tool = SE3.Rx(pi)

    return robot, cyl, stick
