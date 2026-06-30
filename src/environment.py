import os, time
import numpy as np

import swift
from spatialmath import SE3
from spatialmath.base import transl
from spatialgeometry import Cuboid, Cylinder, Sphere, Mesh
import threading
import roboticstoolbox as rtb
from math import pi
from ir_support import CylindricalDHRobotPlot
from ABB_environment import build_ABB_environment
from GP7_environment_fixed import build_GP7_environment
# ⚠️From lab 5 collision
from itertools import combinations
from spatialmath.base import transl
from spatialgeometry import Sphere
from ir_support import line_plane_intersection, RectangularPrism
import numpy as np
import numpy.linalg as npl

# ⚠️helper functions from lab 5 for collision detection

#Explains whether a line-plane intersection point is inside a triangle using barycentric coordinates
#Barycentric coordinates: is when you express a point as a linear combination of the triangle's vertices
# i.e. a point is inside the triangle if it can be expressed as P = A + s*(B - A) + t*(C - A) where 0 <= s,t and s + t <= 1
# P = intersection point, A = vertex 1, B = vertex 2, C = vertex 3, t = barycentric coordinate 1, s = barycentric coordinate 2, s + t = 1
#On Canvas: Lab5 line-plane intersections
def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    """
    Check if a point is inside a triangle in 3D space using barycentric coordinates.
    - param intersect_p: The intersection point (numpy array of shape (3,)).
    - param triangle_verts: The vertices of the triangle (numpy array of shape (3, 3)).
    - return: True if the point is inside the triangle, False otherwise.
    """
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]
    uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    w = intersect_p - triangle_verts[0, :]
    wu, wv = np.dot(w, u), np.dot(w, v)
    D = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:
        return False
    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:
        return False
    return True

# Returns the homogeneous transforms of every link at configuration q via fkine_all()
# Forward kinematics (FK), and homogeneous transforms
def get_link_poses(robot, q=None):
    """
    Get the transformation matrices of each link in the robot.
    - param robot: The robot model (instance of DHRobot or similar).
    - param q: The joint angles (list or numpy array). If None, use the robot's current joint angles.
    - return: A list of transformation matrices (numpy arrays) for each link.
    """
    return robot.fkine_all(q).A if q is not None else robot.fkine_all().A

# Sweeps a trajectory (q_matrix) and checks each consecutive link segment against every triangular face
# using the line-plane intersection + inside-triangle test
# Inside-triangle test: This is done using barycentric coordinates to determine if the intersection point lies within the triangle's boundaries.
def is_collision(robot, q_matrix, faces, vertex, face_normals, env=None):
    """
    Checks whether any link segment of the robot collides with the given object surface.
    Uses line-plane intersection + inside-triangle test from Lab 5.
    """
    for q in q_matrix:
        # Get link poses (list of 4×4 matrices)
        transforms = get_link_poses(robot, q)

        # Iterate through consecutive link pairs
        for i in range(len(transforms) - 1):
            p1 = transforms[i][:3, 3]
            p2 = transforms[i + 1][:3, 3]

            # For every face of the obstacle
            for j, face in enumerate(faces):
                vert_on_plane = vertex[face][0]
                normal = face_normals[j]

                # Check line–plane intersection
                intersect_p, check = line_plane_intersection(normal, vert_on_plane, p1, p2)
                if check == 1:
                    # If intersection exists, check if it’s inside the face polygon
                    triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertex[triangle]):
                            # Add a red marker for visualization
                            if env is not None:
                                marker = Sphere(radius=0.015, color=[1, 0, 0, 1])
                                marker.T = SE3(intersect_p[0], intersect_p[1], intersect_p[2])
                                env.add(marker)
                            return True  # ✅ collision detected
    return False  # ✅ no collision

#This was an old function, no longer used as it was replaced by the CollisionManager class
# def check_all_collisions(robot, q, env=None):
#     """Return list of collider names the robot hits at configuration q."""
#     hits = []
#     for name, (verts, faces, norms) in COLLIDERS.items():
#         if is_collision(robot, [q], faces, verts, norms, env=env):
#             hits.append(name)
#     return hits


class CollisionManager:
    """
    General-purpose collision checker usable by any 6-DoF DHRobot.
    Builds simple Cuboid colliders and runs Lab-5 style link–plane tests.
    """
# Initializes the CollisionManager with an environment to visualize colliders
    def __init__(self, env):
        from ir_support import RectangularPrism
        self.env = env
        self.colliders = {}  # {name: (verts, faces, norms)}
# Adds a box collider to the manager and visualizes it in the environment
    def add_box(self, name, center, half_extents, color=[0,1,0,0.2]):
        """Register and visualize a collision box."""
        from spatialgeometry import Cuboid
        from spatialmath import SE3
        v, f, n = RectangularPrism(*half_extents, center=center).get_data()
        self.colliders[name] = (v, f, n)
        self.env.add(Cuboid(scale=2*np.array(half_extents),
                            pose=SE3(*center),
                            color=color))
# Clears all registered colliders
    def clear(self):
        self.colliders.clear()
# Checks for collisions at configuration q and returns a list of hit colliders
# The parameters are:
# robot: The robot model (instance of DHRobot or similar).
# q: The joint configuration to check for collisions.
# verts: The vertices of the collider's mesh.
# faces: The faces of the collider's mesh.
# norms: The normals of the collider's mesh faces.
    def check(self, robot, q):
        """Return list of colliders hit at configuration q."""
        hits = []
        transforms = robot.fkine_all(q).A
        for name, (verts, faces, norms) in self.colliders.items():
            for i in range(len(transforms) - 1):
                p1, p2 = transforms[i][:3,3], transforms[i+1][:3,3]
                for j, face in enumerate(faces):
                    vert_on_plane = verts[face][0]
                    normal = norms[j]
                    ip, ok = line_plane_intersection(normal, vert_on_plane, p1, p2)
                    if ok == 1:
                        tri_list = np.array(list(combinations(face, 3)), dtype=int)
                        for tri in tri_list:
                            if is_intersection_point_inside_triangle(ip, verts[tri]):
                                hits.append(name)
                                # ensure Python float types
                                ip = np.asarray(ip, dtype=float).flatten()
                                self.env.add(Sphere(radius=0.012,
                                                    pose=SE3(float(ip[0]), float(ip[1]), float(ip[2])),
                                                    color=[1.0, 0.0, 0.0, 1.0]))
                                break
                if hits: break
        return hits

# ⚠️

# Global safety gate 
class Safety:
    # Manages E-STOP and motion gating
    # Thread-safe events are used to signal E-STOP state and whether motion is allowed
    def __init__(self):
        import threading
        self.estop = threading.Event() # set when e-stop is active
        self.run_gate = threading.Event()  # set when motion may proceed
        self.run_gate.set()  # start in "running" state
        self.can_resume = False # requires explicit resume after reset
# Trigger an emergency stop: set estop and block motion
    def estop_now(self):
        self.estop.set()
        self.run_gate.clear()
        self.can_resume = False
# Reset the E-STOP: clear estop but keep motion blocked until resume() is called
    def reset(self):
        # disarm e-stop but keep motion blocked until resume()
        self.estop.clear()
        self.can_resume = True
# Resume motion if allowed
    def resume(self):
        if not self.estop.is_set() and self.can_resume:
            self.run_gate.set()
            self.can_resume = False
            return True
        return False
# Block until motion is allowed
    def wait_ok(self):  # non-busy block until allowed to move
        self.run_gate.wait() # returns immediately if already set
# Global safety instance
SAFETY = Safety()

# ---- Arduino hardware E-Stop bridge -----------------------------------------
import threading as _th, time as _time
import importlib, importlib.util as _il_util
if _il_util.find_spec("serial") is not None:
    _ser = importlib.import_module("serial")
else:
    _ser = None

_arduino = None  # will hold the opened serial port
# Physical hardware E-STOP listener
# Plug in an Arduino to COM3 (or other port) running the simple E-STOP sketch
# rx loop listens for 'E' and triggers SAFETY.estop_now()
# threading is used to run the listener in the background so it doesn't block the main program
def start_hw_estop_bridge(port="COM3", baud=115200):
    """Listen for 'E' from Arduino and trigger SAFETY.estop_now().
       We also provide _hw_echo() so Python can send 'R'/'G' back to the board.
    """
    global _arduino
    if _ser is None:
        print("[HW] pyserial not installed. Run: pip install pyserial")
        return None
    try:
        _arduino = _ser.Serial(port=port, baudrate=baud, timeout=0.1)
        print(f"[HW] Arduino E-Stop connected on {port}")
    except Exception as e:
        print(f"[HW] Could not open {port}: {e}")
        _arduino = None
        return None

    def _rx_loop():
        while True:
            try:
                ln = _arduino.readline().strip()
                if not ln:
                    _time.sleep(0.01); continue
                msg = ln.decode(errors="ignore")
                if msg == "E":
                    SAFETY.estop_now()
                    print("[HW] E-STOP from hardware button")
            except Exception:
                _time.sleep(0.1)

    _th.Thread(target=_rx_loop, daemon=True).start()
    return _arduino

# After estop is triggered, this function sends 'R' to the Arduino to update the LED status
def _hw_echo(cmd: str):
    """Send 'R' or 'G' so the Arduino LEDs reflect reset/resume actions."""
    try:
        if _arduino:
            _arduino.write((cmd + "\n").encode())
    except Exception:
        pass
# -----------------------------------------------------------------------------

# The safety light curtain class
# Rectangular curtain (AABB) in WORLD frame, if any link enters the curtain, we 'hold' motion (clear SAFETY.run_gate) but do NOT set ESTOP.
# Reopens when clear for a short debounce
class LightCurtain:
    """
    Rectangular curtain (AABB) in WORLD frame.
    - When any robot link origin enters the curtain, we 'hold' motion
      (clear SAFETY.run_gate) but do NOT set ESTOP.
    - When clear for a short debounce, we reopen run_gate automatically.
    """
    # Initializes the LightCurtain with its pose, size, environment, and debounce settings
    # Debounce is used to prevent rapid toggling of the hold state
    def __init__(self, pose: SE3, size_xyz, env,
                 name="LC",
                 beam_rows=7, beam_cols=18,
                 debounce_enter=0.05, debounce_clear=0.25):
        self.pose = pose
        self.size = np.array(size_xyz, dtype=float)
        self.env = env
        self.name = name

        # visuals: two posts + a grid of small spheres + faint box
        sx, sy, sz = self.size
        post_r = 0.02
        left  = pose * SE3(-sx/2, -sy/2, 0)
        right = pose * SE3(-sx/2, +sy/2, 0)

        self._objects = []
        self._objects.append(Cylinder(radius=post_r, length=sz, pose=left,  color=[0.2,0.2,0.2,1.0]))
        self._objects.append(Cylinder(radius=post_r, length=sz, pose=right, color=[0.2,0.2,0.2,1.0]))

        for zi in np.linspace(-sz/2+0.02, sz/2-0.02, beam_rows):
            for yi in np.linspace(-sy/2, sy/2, beam_cols):
                p = pose * SE3(0, yi, zi)
                self._objects.append(Sphere(radius=0.006, pose=p, color=[1.0, 0.0, 0.0, 0.75]))

        self._aabb_viz = Cuboid(scale=self.size, pose=self.pose, color=[1.0, 0.2, 0.2, 0.08])
        self._objects.append(self._aabb_viz)

        self._visible = True
        self._thread = None
        self._stop = threading.Event()
        self._debounce_enter = debounce_enter
        self._debounce_clear = debounce_clear
        self._last_state = False  # False = clear, True = broken
        self._t_edge = time.time()

    # Adds the light curtain objects to the environment
    def add_to_env(self):
        for o in self._objects:
            self.env.add(o)

    # Sets the visibility of the light curtain objects
    def set_visible(self, on: bool):
        self._visible = bool(on)
        alpha = 1.0 if on else 0.0
        for o in self._objects:
            try:
                c = o.color
                o.color = [c[0], c[1], c[2], alpha]
            except Exception:
                pass
        try: self.env.step(0)
        except Exception: pass

    # Checks if a point in world coordinates is within the local AABB of the light curtain
    # This is done by transforming the point into the local frame of the curtain and checking its coordinates against the half-extents
    def _within_local_aabb(self, p_world: np.ndarray) -> bool:
        Ainv = np.linalg.inv(self.pose.A)
        ph = np.r_[p_world, 1.0]
        pl = Ainv @ ph
        x, y, z = pl[:3]
        hx, hy, hz = self.size / 2.0
        return (-hx <= x <= hx) and (-hy <= y <= hy) and (-hz <= z <= hz)

    # Checks if any link of the robot is within the light curtain's AABB
    def _any_link_in_box(self, robot) -> bool:
        try:
            As = robot.fkine_all(robot.q).A
        except Exception:
            return False
        for i in range(len(As) -1):
            p1 = As[i][:3, 3]
            p2 = As[i + 1][:3, 3]
            # check several points along the link
            for alpha in np.linspace(0, 1, 5):
                p_link = (1 - alpha) * p1 + alpha * p2
                if self._within_local_aabb(p_link):
                    return True
        # for A in As:
        #     if self._within_local_aabb(A[:3, 3]):
        #         return True
        return False

    # If hold is True, motion is blocked; if False, motion is allowed
    def _set_hold(self, hold: bool):
        # do not override an active ESTOP
        if SAFETY.estop.is_set():
            return
        if hold:
            SAFETY.run_gate.clear()
        else:
            SAFETY.run_gate.set()

    # Starts monitoring the robots for light curtain breaches
    def start_monitor(self, robots: dict, status_label=None, poll=0.02):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Monitoring loop
        # This is the main loop that checks the state of the light curtain
        def loop():
            while not self._stop.is_set():
                broken = any(self._any_link_in_box(bot) for bot in robots.values())
                t = time.time()
                if broken != self._last_state:
                    need = self._debounce_enter if broken else self._debounce_clear
                    if (t - self._t_edge) >= need:
                        self._last_state = broken
                        self._t_edge = t
                        if broken:
                            self._set_hold(True)
                            if status_label: status_label.desc = "Light curtain BROKEN — motion held"
                            print("[LC] Beam broken — holding motion.")
                        else:
                            self._set_hold(False)
                            if status_label: status_label.desc = "Light curtain CLEAR — motion allowed"
                            print("[LC] Beam clear — motion allowed.")
                time.sleep(poll)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        print(f"[LC] monitoring {len(robots)} robots")

    # Stops the light curtain monitoring thread
    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=0.5)


#  Minimal shared GUI for this module
# Sets up a GUI with joint sliders and a robot selector
# This function creates a GUI with joint sliders and a robot selector for multiple robots.
# def _set_status(env, label, text): this is used to update the status label in the GUI
def _set_status(env, label, text):
    try: label.desc = text
    except Exception: pass

# Builds a GUI with joint sliders and a robot selector for multiple robots
def build_gui_multi(env, robots: dict, default_robot: str = None):
    """Single bank of 6 joint sliders + a robot selector (dropdown or buttons)."""
    if not robots:
        print("[GUI] No robots provided"); return
    # Whatever robot is active right now will be controlled by the sliders
    active_name = default_robot if (default_robot in robots) else next(iter(robots.keys()))
    active = robots[active_name]

    # Status label: Set the corresponding name of the active robot
    status_label = swift.Label(f"Controlling: {active_name}")
    env.add(status_label)

    #  Selector (dropdown if available, else buttons) 
    # This is simply used to switch between different robots in the GUI
    def on_select(name: str):
        nonlocal active_name, active
        if name in robots:
            active_name, active = name, robots[name]
            _set_status(env, status_label, f"Controlling: {active_name}")
            # sync slider values to the newly selected robot
            q_now = active.q if active.q is not None else getattr(active, "qz", np.zeros(active.n))
            for i, s in enumerate(sliders[:min(6, active.n)]):
                try: s.value = float(q_now[i])
                except Exception: pass

    # selector_added is a flag to check if the selector was successfully added
    # the selector is a dropdown menu that allows the user to select which robot to control
    selector_added = False
    try:
        sel = swift.Select(
            options=list(robots.keys()),
            value=active_name,
            desc="Robot",
            cb=lambda v: on_select(str(v)),
        )
        env.add(sel)
        selector_added = True
    except Exception:
        pass

    if not selector_added:
        # radio-style buttons fallback
        for name in robots.keys():
            try:
                env.add(swift.Button(desc=f"Control {name}",
                                     cb=lambda _=None, n=name: on_select(n)))
            except Exception:
                pass

    #One bank of 6 sliders bound to the currently selected robot 
    sliders = []
    q_now = active.q if active.q is not None else getattr(active, "qz", np.zeros(active.n))
    for i in range(min(6, active.n)):
        def _cb(val, j=i):
            if SAFETY.estop.is_set():
                print("Motion blocked by safety (E-STOP)."); return
            SAFETY.wait_ok()

    # update whichever robot is active right now
            q = active.q if active.q is not None else getattr(active, "qz", np.zeros(active.n))
            q = q.copy(); q[j] = float(val)

    # Predictive collision test at the *next* pose
            hits = collision_mgr.check(active, q)
            if hits:
                print(f"[LIVE] ⚠️ {active_name} collision: {', '.join(hits)}")
                SAFETY.estop_now()
                return
    # commit
            active.q = q
            try: env.step(0)
            except Exception: pass
            _set_status(env, status_label, f"{active_name}: J{j+1} = {val:.3f}")
        s = swift.Slider(cb=_cb, min=-3.14, max=3.14, step=0.01,
                         value=float(q_now[i]), desc=f"Joint {i+1}", unit="rad")
        env.add(s)
        sliders.append(s)

    print("[GUI] Shared sliders + robot selector ready.")

    def kb_loop():
        help_text = (
            "\n[Keyboard]\n"
            "  help\n"
            "  estop | reset | resume\n"
            "  lcon  (toggle light curtain visual if available)\n"
            "  home  (active robot -> qz)\n"
            "  x+/x-, y+/y-, z+/z- (cartesian jog 2cm)\n"
            "  use <name>  (switch active robot: e.g., 'use ur3')\n"
        )
        print(help_text)

        # simple safety flags for module-level path
        E_STOP = False
        COLLISION_HOLD = False
        can_resume = False

        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue
            cmd = line.lower()

            if cmd == "help":
                print(help_text)
                continue

            if cmd == "estop":
                SAFETY.estop_now()
                _hw_echo("E") # tell Arduino to light E-STOP LED
                _set_status(env, status_label, "E-STOP ACTIVE")
                print("Emergency stop pressed.")
                continue

            if cmd == "reset":
                SAFETY.reset()
                _hw_echo("R") # tell Arduino to light RESET LED
                _set_status(env, status_label, "Reset done — press 'resume'")
                print("E-stop acknowledged. System is reset; press Resume to continue.")
                continue

            if cmd == "resume":
                if SAFETY.resume():
                    _hw_echo("G") # tell Arduino to light GO LED
                    _set_status(env, status_label, "Running")
                    print("System resumed from E-stop.")
                else:
                    print("Can't resume (did you 'reset' first, and is E-STOP cleared?).")
                continue
            
            if cmd == "lcon":
                try:
                    LIGHT_CURTAIN.set_visible(not LIGHT_CURTAIN._visible)
                    print(f"[LC] visibility -> {LIGHT_CURTAIN._visible}")
                except Exception as e:
                    print("[LC] not initialised", e)
                continue

            if cmd == "lcrestart":
                try:
                    LIGHT_CURTAIN.start_monitor(robots=robots, status_label=None)
                    print("[LC] monitor restarted")
                except Exception as e:
                    print("[LC] restart error:", e)
                continue

            if cmd == "home":
                # home the currently selected robot
                qz = getattr(active, "qz", None)
                if qz is not None:
                    active.q = qz
                    # resync sliders
                    q_now = active.q if active.q is not None else qz
                    for i, s in enumerate(sliders[:min(6, active.n)]):
                        try: s.value = float(q_now[i])
                        except Exception: pass
                    try: env.step(0)
                    except Exception: pass
                    _set_status(env, status_label, "Homed (qz)")
                else:
                    print("[home] Active robot has no qz")
                continue


            if cmd == "lcstatus":
                try:
                    print(f"[LC] visible={LIGHT_CURTAIN._visible}  broken={LIGHT_CURTAIN._last_state}  "
                    f"estop={SAFETY.estop.is_set()}  run_gate={SAFETY.run_gate.is_set()}")
                except NameError:
                    print("[LC] not initialised")
                continue

            if cmd == "lchold":
                if not SAFETY.estop.is_set():
                    SAFETY.run_gate.clear()
                    print("[LC] forced HOLD (run_gate cleared)")
                else:
                    print("[LC] estop active; cannot force hold")
                continue

            if cmd == "lcrelease":
                if not SAFETY.estop.is_set():
                    SAFETY.run_gate.set()
                    print("[LC] forced RELEASE (run_gate set)")
                else:
                    print("[LC] estop active; cannot release")
                continue

            if cmd == "lcoff":
                try:
                    LIGHT_CURTAIN.stop()
                    print("[LC] monitor stopped (curtain visuals may still be visible)")
                except Exception as e:
                    print("[LC] stop error:", e)
                continue

            # Switch active robot: "use ur3" / "use kuka" / etc.
            if cmd.startswith("use "):
                candidate = cmd.split(" ", 1)[1].strip()
                if candidate in robots:
                    active_name, active = candidate, robots[candidate]
                    _set_status(env, status_label, f"Controlling: {active_name}")
                    # sync slider values to the newly selected robot
                    q_now = active.q if active.q is not None else getattr(active, "qz", np.zeros(active.n))
                    for i, s in enumerate(sliders[:min(6, active.n)]):
                        try: s.value = float(q_now[i])
                        except Exception: pass
                    print(f"[use] Switched to {active_name}")
                else:
                    print(f"[use] Robot '{candidate}' not found. Options: {list(robots.keys())}")
                continue

            # Cartesian jogs (2 cm per key) with simple IK
            jog_map = {
                "x+": (0.02, 0.00, 0.00), "x-": (-0.02, 0.00, 0.00),
                "y+": (0.00, 0.02, 0.00), "y-": (0.00, -0.02, 0.00),
                "z+": (0.00, 0.00, 0.02), "z-": (0.00, 0.00, -0.02),
            }
            if cmd in jog_map:
                if SAFETY.estop.is_set():
                    print("Motion blocked by safety.")
                    continue
                SAFETY.wait_ok() 
                dx, dy, dz = jog_map[cmd]
                try:
                    T_now = active.fkine(active.q)
                    T_goal = SE3(T_now) * SE3.Trans(dx, dy, dz)
                    sol = active.ikine_LM(T_goal, q0=active.q, ilimit=100, tol=1e-4)
                    if getattr(sol, "success", False):
                        active.q = sol.q
                        q_now = active.q if active.q is not None else getattr(active, "qz", np.zeros(active.n))
                        for i, s in enumerate(sliders[:min(6, active.n)]):
                            try: s.value = float(q_now[i])
                            except Exception: pass
                        try: env.step(0)
                        except Exception: pass
                        _set_status(env, status_label, f"{active_name}: jog {cmd} OK")
                    else:
                        _set_status(env, status_label, f"{active_name}: jog {cmd} IK failed")
                except Exception as e:
                    _set_status(env, status_label, f"Jog {cmd} error: {e}")
                continue

            print(f"Unknown command '{cmd}'. Type 'help'.")

    threading.Thread(target=kb_loop, daemon=True).start()
    print("[GUI] Keyboard controls ready (type 'help' in the terminal).")

# ---- Environment constants and helper functions ------------------------------
DT = 1/60.0
STEPS_FAST, STEPS_MED, STEPS_SLOW = 30, 50, 80


TABLE_CENTER = np.array([0.55, 0.00, 0.16])
TABLE_HALF   = np.array([0.14, 0.10, 0.14])
TABLE_TOP_Z  = float(TABLE_CENTER[2] + TABLE_HALF[2])


SPOON_LEN, SPOON_RAD = 0.16, 0.011
POT_R = 0.08

# Checks if any part of the robot is below the table top plus a small margin
# This is done by checking the minimum z-coordinate of all link transforms
# In English: This function checks if any part of the robot is below the table top plus a small margin (0.02 meters).
def robot_hits_table(robot, q):
    try:
        Ts = robot.fkine_all(q).A
        min_z = min(T[2,3] for T in Ts)
        return min_z < TABLE_TOP_Z + 0.02
    except Exception:
        return False

# Safely executes a trajectory while checking for collisions and handling E-STOP
# If a spoon is attached, it updates the spoon's pose based on the robot's end-effector pose
# The parameters are:
# robot: The robot model (instance of DHRobot or similar).
# traj: A list of joint configurations (numpy arrays) representing the trajectory.
# spoon_attached: A boolean indicating if a spoon is attached to the robot.
# spoon_obj: The spoon object to update the pose of (if attached).
# T_rel: The relative transform from the robot's end-effector to the spoon (if attached).

#How this checks for collisions:
# For each configuration in the trajectory, it checks for collisions using the COLLISION MANAGER.
# If a collision is detected, it triggers an E-STOP and halts the trajectory execution
def safe_execute_trajectory(robot, traj, spoon_attached=False, spoon_obj=None, T_rel=None):
    for q in traj:
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: trajectory halted")
            return robot.q

    # hard collision test against all scene colliders
        hits = collision_mgr.check(robot, q)
        if hits:
            print(f"[COLLISION] {robot.name} hit {hits}")
            SAFETY.estop_now()
            return robot.q

        robot.q = q
        if spoon_attached and spoon_obj is not None and T_rel is not None:
            try:
                Tee = robot.fkine(q)
                spoon_obj.T = (Tee * T_rel).A
            except Exception:
                pass
        env.step(DT)
    return robot.q   # <-- moved outside the loop

#UR3 
# Creates and returns a UR3 robot model with a specific base pose and joint configuration
def make_ur3():
    bot = rtb.models.UR3()
    bot.base = SE3(0.25, 0.0, 0.0)
    bot.q = np.array([0.0, -np.pi/3, np.pi/3, -np.pi/2, -np.pi/2, 0.0])
    return bot

# kitchen
def build_kitchen(env):
    # === Fire extinguisher STL ===
    assets_dir = os.path.join(os.path.dirname(__file__), "Main_Meshes")
    fire_path  = os.path.join(assets_dir, "Fire_scaled_meters_base.stl")

    if os.path.exists(fire_path):
    
        FIRE_BASE_POSE = SE3(-1.9, -2.1, 0.02) * SE3.Rz(0.6)
        fire_base = Mesh(
            filename=fire_path,
            pose=FIRE_BASE_POSE,
            scale=[1.0, 1.0, 1.0],          # units already in meters
            color=[1.0, 0.0, 0.0, 1.0]      # optional tint
    )
        env.add(fire_base)
    else:
        print(f"[MESH] Missing file: {fire_path}")

    # Floor
    env.add(Cuboid(scale=[6.0, 6.0, 0.02], pose=SE3(0, 0, -0.03), color=[0.40, 0.40, 0.43, 1]))

    # Back counter
    counter_h = 0.10
    back_ctr_center = np.array([0.95, -0.15, counter_h/2])
    back_ctr_half   = np.array([0.60, 0.50, counter_h/2])
    back_ctr = Cuboid(scale=2*back_ctr_half, pose=SE3(*back_ctr_center), color=[0.75, 0.75, 0.78, 0.95])
    env.add(back_ctr)

    # Stove top
    burner_R = 0.06
    stove_top_z = back_ctr_center[2] + back_ctr_half[2]
    burner_centers = [
        np.array([back_ctr_center[0]-0.20, back_ctr_center[1]-0.10, stove_top_z+0.005]),
        np.array([back_ctr_center[0]-0.20, back_ctr_center[1]+0.10, stove_top_z+0.005]),
        np.array([back_ctr_center[0]+0.00, back_ctr_center[1]-0.10, stove_top_z+0.005]),
        np.array([back_ctr_center[0]+0.00, back_ctr_center[1]+0.10, stove_top_z+0.005]),
    ]
    for c in burner_centers:
        for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
            rim = c + np.array([burner_R*np.cos(a), burner_R*np.sin(a), 0.0])
            env.add(Sphere(radius=0.008, pose=SE3(*rim), color=[0.15, 0.15, 0.15, 1]))

    # benches
    audience_counter_h = 0.10
    audience_half = np.array([0.35, 0.25, audience_counter_h/2])
    robot_x = 0.25
    audience_centers = [
        np.array([robot_x - 1.2, -0.8, audience_counter_h/2]),
        np.array([robot_x - 1.2,  0.0, audience_counter_h/2]),
        np.array([robot_x - 1.2,  0.8, audience_counter_h/2]),
    ]
    for ctr in audience_centers:
        env.add(Cuboid(scale=2*audience_half, pose=SE3(*ctr), color=[0.65, 0.65, 0.68, 0.95]))
        top_z = ctr[2] + audience_half[2]
        burner_offsets = [
            np.array([-0.08, -0.08, 0.005]),
            np.array([-0.08,  0.08, 0.005]),
            np.array([ 0.08, -0.08, 0.005]),
            np.array([ 0.08,  0.08, 0.005]),
        ]
        for off in burner_offsets:
            c = ctr + np.array([0, 0, audience_half[2]]) + off
            for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
                rim = c + np.array([burner_R*np.cos(a), burner_R*np.sin(a), 0.0])
                env.add(Sphere(radius=0.008, pose=SE3(*rim), color=[0.15, 0.15, 0.15, 1]))

    # Walls
    wall_h, wall_t = 2.5, 0.1
    env.add(Cuboid(scale=[6.0, wall_t, wall_h], pose=SE3(0, 3.0, wall_h/2), color=[0.8, 0.8, 0.85, 0.9]))
    env.add(Cuboid(scale=[wall_t, 6.0, wall_h], pose=SE3(-3.0, 0, wall_h/2), color=[0.8, 0.8, 0.85, 0.9]))
    env.add(Cuboid(scale=[wall_t, 6.0, wall_h], pose=SE3( 3.0, 0, wall_h/2), color=[0.8, 0.8, 0.85, 0.9]))
    env.add(Cuboid(scale=[2.0, wall_t, wall_h], pose=SE3(-1.0, -3.0, wall_h/2), color=[0.8, 0.8, 0.85, 0.9]))
    env.add(Cuboid(scale=[2.0, wall_t, wall_h], pose=SE3( 1.0, -3.0, wall_h/2), color=[0.8, 0.8, 0.85, 0.9]))

    # Upper cabinets
    cabinet_h = 0.8
    cabinet_z = wall_h - cabinet_h/2 - 0.3
    env.add(Cuboid(scale=[1.5, 0.3, cabinet_h], pose=SE3(-1.0, 2.7, cabinet_z), color=[0.6, 0.4, 0.2, 0.8]))
    env.add(Cuboid(scale=[1.5, 0.3, cabinet_h], pose=SE3( 1.0, 2.7, cabinet_z), color=[0.6, 0.4, 0.2, 0.8]))
    
import numpy as np
from scipy import linalg
from spatialmath.base import rpy2r, tr2rpy
from spatialmath import SE3

# RMRC is the Resolved Motion Rate Control algorithm
# What it does is:
# 1. Computes the desired end-effector pose based on the task (e.g., stirring)
# 2. Uses inverse kinematics to compute the joint configurations that achieve this pose
# 3. Applies a motion model to smoothly move the robot along the desired trajectory

# For GP7 with RMRC, this does the above 3 steps by:
# - Moving to pick up the spoon
# - Moving above the pot
# - Lowering into the pot
# - Stirring in a circular motion while applying a wrist sine wave for realism

#Specific parameters:
# - stir_R: Radius of the stirring circle
# - tip_depth: Depth of the spoon tip in the pot
# - seconds: Total time to stir
# - delta_t: Time step for each motion update
# - revs: Number of revolutions to stir
# - wrist_sine_amp_deg: Amplitude of the wrist sine wave in degrees
# - max_joint_vel: Maximum joint velocity for safety

# The motion model in this is the code line: robot.q = qnext
# This line updates the robot's joint configuration to the next computed configuration (qnext) at each time step.

# RMRC is used to make it smoother and more realistic, rather than jumping directly to target poses.
# Jtraj vs RMRC: Jtraj is a simple joint space trajectory generator, while RMRC uses task space control for smoother motion.

def rmrc_stir_GP7_smooth(env, robot, cyl, stick_obj, base_T, pot_center_local,
                         stir_R=0.085, tip_depth=0.02, seconds=20.0, delta_t=0.05,
                         revs=3, wrist_sine_amp_deg=20, max_joint_vel=1.5):

    print("Starting GP7 simple stirring...")

    # ADDED: Spoon attachment variables
    grabbed = False
    T_spoon_rel = None
    
    def update_spoon():
        """Keep spoon attached to end effector when grabbed"""
        if grabbed and stick_obj is not None and T_spoon_rel is not None:
            try:
                stick_obj.T = (robot.fkine(robot.q) * T_spoon_rel).A
            except Exception:
                pass

    # --- Local -> World helpers
    def TL2W(x, y, z):
        return base_T * SE3(x, y, z)  # world pose from local xyz

    potL = pot_center_local.copy()
    safe_zL = float(potL[2] + 0.20)
    stir_zL = float(potL[2] + tip_depth)

    def step_q(qnext):
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: GP7 stirring halted"); return False
        hits = collision_mgr.check(robot, qnext)
        if hits:
            print(f"GP7 collision detected: {hits}")
            SAFETY.estop_now(); return False
        robot.q = qnext
        cyl.update(qnext)
        update_spoon()  # ADDED: Update spoon position
        env.step(delta_t)
        return True

    def ik_pos_keepseed_L(xL, yL, zL, qseed):
        T_W = TL2W(xL, yL, zL)  # convert local target to WORLD before IK
        sol = robot.ikine_LM(T_W, q0=qseed, mask=[1,1,1,0,0,0])
        return sol.q if getattr(sol, "success", False) else qseed

    # ADDED: Step 1 - Move to spoon and grab it
    print("GP7: Moving to spoon...")
    spoon_center_local = (base_T.inv() * SE3(stick_obj.T)).t.reshape(-1)
    pre_spoon = np.array([spoon_center_local[0], spoon_center_local[1], safe_zL])
    pick_z = max(spoon_center_local[2] + 0.01, 0.07)
    
    # Approach spoon
    q_pre_spoon = ik_pos_keepseed_L(*pre_spoon, robot.q)
    for q in rtb.jtraj(robot.q, q_pre_spoon, 30).q:
        if not step_q(q): break
    
    # Lower to spoon
    q_at_spoon = ik_pos_keepseed_L(spoon_center_local[0], spoon_center_local[1], pick_z, robot.q)
    for q in rtb.jtraj(robot.q, q_at_spoon, 20).q:
        if not step_q(q): break
    
    # GRAB SPOON: Calculate relative transform (same as UR3 and KUKA)
    print("GP7: Grabbing spoon...")
    Tee = robot.fkine(robot.q)
    T_spoon_rel = Tee.inv() * SE3(stick_obj.T)
    grabbed = True
    update_spoon()
    
    # Lift spoon
    q_lift_spoon = ik_pos_keepseed_L(spoon_center_local[0], spoon_center_local[1], safe_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_lift_spoon, 20).q:
        if not step_q(q): break

    # --- Step 2 - Approach above the pot (WORLD IK from LOCAL point)
    print("GP7: Moving to pot...")
    pre_potL = np.array([potL[0], potL[1], safe_zL])
    q_pre = ik_pos_keepseed_L(*pre_potL, robot.q)
    for q in rtb.jtraj(robot.q, q_pre, 40).q:  # Slower movement to pot
        if not step_q(q): break

    # --- Lower to stir height
    print("GP7: Lowering into pot...")
    q_stir = ik_pos_keepseed_L(potL[0], potL[1], stir_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_stir, 20).q:
        if not step_q(q): break

    # --- Stirring loop (LOCAL circle → WORLD IK each segment)
    print("GP7: Stirring with attached spoon...")
    R = float(stir_R); steps_per_rev = 45; seg_steps = 2
    wrist_amp = np.deg2rad(wrist_sine_amp_deg)
    q_prev = robot.q.copy()

    for k in range(revs * steps_per_rev):
        if k % 5 == 0:  # Check safety every 5th step for speed
            SAFETY.wait_ok()
            if SAFETY.estop.is_set():
                print("E-STOP: GP7 stirring halted")
                break
                
        th = 2*np.pi*k/steps_per_rev
        xL = potL[0] + R*np.cos(th)
        yL = potL[1] + R*np.sin(th)
        q_tgt = ik_pos_keepseed_L(xL, yL, stir_zL, q_prev)
        q_tgt[-1] += wrist_amp * np.sin(2*th)

        for qk in rtb.jtraj(q_prev, q_tgt, seg_steps).q:
            if not step_q(qk): break
            
        # Add blue markers for GP7's stirring path
        if k % 10 == 0:
            env.add(Sphere(radius=0.008, pose=SE3(robot.fkine(robot.q).t), color=[0.0, 0.0, 1.0, 0.55]))
            
        q_prev = q_tgt.copy()

    # --- Lift out
    print("GP7: Lifting out...")
    q_lift = ik_pos_keepseed_L(potL[0], potL[1], safe_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_lift, 20).q:
        if not step_q(q): break

    print("GP7 simple stirring complete!")

def rmrc_stir_ABB_with_pickup(env, robot, cyl, spoon_obj, base_T, pot_center_local,
                              stir_R=0.085, tip_depth=0.02, seconds=20.0, delta_t=0.05,
                              revs=3, wrist_sine_amp_deg=20, max_joint_vel=1.5):
    """
    ABB stirring with spoon pickup first, then RMRC stirring (same approach as GP7).
    """
    print("Starting ABB with spoon pickup...")

    # Spoon attachment variables
    grabbed = False
    T_spoon_rel = None
    
    def update_spoon():
        """Keep spoon attached to end effector when grabbed"""
        if grabbed and spoon_obj is not None and T_spoon_rel is not None:
            try:
                spoon_obj.T = (robot.fkine(robot.q) * T_spoon_rel).A
            except Exception:
                pass

    # --- Local -> World helpers
    def TL2W(x, y, z):
        return base_T * SE3(x, y, z)  # world pose from local xyz

    potL = pot_center_local.copy()
    safe_zL = float(potL[2] + 0.20)
    stir_zL = float(potL[2] + tip_depth)

    def step_q(qnext):
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: ABB stirring halted"); return False
        hits = collision_mgr.check(robot, qnext)
        if hits:
            print(f"ABB collision detected: {hits}")
            SAFETY.estop_now(); return False
        robot.q = qnext
        cyl.update(qnext)
        update_spoon()  
        env.step(delta_t)
        return True

    def ik_pos_keepseed_L(xL, yL, zL, qseed):
        T_W = TL2W(xL, yL, zL)  # convert local target to WORLD before IK
        sol = robot.ikine_LM(T_W, q0=qseed, mask=[1,1,1,0,0,0])
        return sol.q if getattr(sol, "success", False) else qseed

    # ADDED: Step 1 - Move to spoon and grab it (same as GP7)
    print("ABB: Moving to spoon...")
    spoon_center_local = (base_T.inv() * SE3(spoon_obj.T)).t.reshape(-1)
    pre_spoon = np.array([spoon_center_local[0], spoon_center_local[1], safe_zL])
    pick_z = max(spoon_center_local[2] + 0.01, 0.07)
    
    # Approach spoon
    q_pre_spoon = ik_pos_keepseed_L(*pre_spoon, robot.q)
    for q in rtb.jtraj(robot.q, q_pre_spoon, 30).q:
        if not step_q(q): break
    
    # Lower to spoon
    q_at_spoon = ik_pos_keepseed_L(spoon_center_local[0], spoon_center_local[1], pick_z, robot.q)
    for q in rtb.jtraj(robot.q, q_at_spoon, 20).q:
        if not step_q(q): break
    
    # GRAB SPOON: Calculate relative transform (same as GP7)
    print("ABB: Grabbing spoon...")
    Tee = robot.fkine(robot.q)
    T_spoon_rel = Tee.inv() * SE3(spoon_obj.T)
    grabbed = True
    update_spoon()
    
    # Lift spoon
    q_lift_spoon = ik_pos_keepseed_L(spoon_center_local[0], spoon_center_local[1], safe_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_lift_spoon, 20).q:
        if not step_q(q): break

    # Step 2 - Move to pot and stir (same as GP7)
    print("ABB: Moving to pot...")
    pre_potL = np.array([potL[0], potL[1], safe_zL])
    q_pre = ik_pos_keepseed_L(*pre_potL, robot.q)
    for q in rtb.jtraj(robot.q, q_pre, 40).q:
        if not step_q(q): break

    print("ABB: Lowering into pot...")
    q_stir = ik_pos_keepseed_L(potL[0], potL[1], stir_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_stir, 20).q:
        if not step_q(q): break

    # Step 3 - Stirring loop with attached spoon
    print("ABB: Stirring with attached spoon...")
    R = float(stir_R); steps_per_rev = 45; seg_steps = 2
    wrist_amp = np.deg2rad(wrist_sine_amp_deg)
    q_prev = robot.q.copy()

    for k in range(revs * steps_per_rev):
        if k % 5 == 0:
            SAFETY.wait_ok()
            if SAFETY.estop.is_set():
                print("E-STOP: ABB stirring halted")
                break
                
        th = 2*np.pi*k/steps_per_rev
        xL = potL[0] + R*np.cos(th)
        yL = potL[1] + R*np.sin(th)
        q_tgt = ik_pos_keepseed_L(xL, yL, stir_zL, q_prev)
        q_tgt[-1] += wrist_amp * np.sin(2*th)

        for qk in rtb.jtraj(q_prev, q_tgt, seg_steps).q:
            if not step_q(qk): break
            
        # Add red markers for ABB's stirring path
        if k % 10 == 0:
            env.add(Sphere(radius=0.008, pose=SE3(robot.fkine(robot.q).t), color=[1.0, 0.0, 0.0, 0.55]))
            
        q_prev = q_tgt.copy()

    # Step 4 - Lift out
    print("ABB: Lifting out...")
    q_lift = ik_pos_keepseed_L(potL[0], potL[1], safe_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_lift, 20).q:
        if not step_q(q): break

    print("ABB stirring with pickup complete!")

def rmrc_stir_KUKA_with_pickup(env, robot, cyl, spoon_obj, base_T, pot_center_local,
                               stir_R=0.085, tip_depth=0.02, seconds=15.0, delta_t=0.04,
                               revs=3, wrist_sine_amp_deg=15, max_joint_vel=1.2):
    """
    KUKA stirring with spoon pickup first - FIXED joint flipping issues.
    """
    print("Starting KUKA with spoon pickup...")

    
    grabbed = False
    T_spoon_rel = None
    
    def update_spoon():
        """Keep spoon attached to end effector when grabbed"""
        if grabbed and spoon_obj is not None and T_spoon_rel is not None:
            try:
                spoon_obj.T = (robot.fkine(robot.q) * T_spoon_rel).A
            except Exception:
                pass

    # --- Local -> World helpers
    def TL2W(x, y, z):
        return base_T * SE3(x, y, z)

    potL = pot_center_local.copy()
    safe_zL = float(potL[2] + 0.20)
    stir_zL = float(potL[2] + tip_depth)

    def step_q(qnext):
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: KUKA stirring halted"); return False
        hits = collision_mgr.check(robot, qnext)
        if hits:
            print(f"KUKA collision detected: {hits}")
            SAFETY.estop_now(); return False
        robot.q = qnext
        cyl.update(qnext)
        update_spoon()
        env.step(delta_t)
        return True

    def ik_pos_keepseed_L(xL, yL, zL, qseed):
        T_W = TL2W(xL, yL, zL)
        sol = robot.ikine_LM(T_W, q0=qseed, mask=[1,1,1,0,0,0])
        return sol.q if getattr(sol, "success", False) else qseed

    # ADDED: Better IK with joint continuity checking
    def ik_pos_smooth(xL, yL, zL, qseed):
        """IK that avoids large joint jumps"""
        T_W = TL2W(xL, yL, zL)
        
        # Try multiple IK solutions and pick the one closest to current config
        best_q = qseed
        best_dist = np.inf
        
        for _ in range(3):  # Try a few different starting points
            sol = robot.ikine_LM(T_W, q0=qseed, mask=[1,1,1,0,0,0])
            if getattr(sol, "success", False):
                # Check joint distance
                joint_dist = np.linalg.norm(sol.q - qseed)
                if joint_dist < best_dist and joint_dist < 2.0:  # Reject large jumps
                    best_q = sol.q
                    best_dist = joint_dist
            
            # Try with slight perturbation
            qseed = qseed + np.random.normal(0, 0.1, len(qseed))
            
        return best_q

    # ADDED: Step 1 - Move to spoon and grab it
    print("KUKA: Moving to spoon...")
    spoon_center_local = (base_T.inv() * SE3(spoon_obj.T)).t.reshape(-1)
    pre_spoon = np.array([spoon_center_local[0], spoon_center_local[1], safe_zL])
    pick_z = max(spoon_center_local[2] + 0.01, 0.07)
    
    # Approach spoon
    q_pre_spoon = ik_pos_keepseed_L(*pre_spoon, robot.q)
    for q in rtb.jtraj(robot.q, q_pre_spoon, 30).q:
        if not step_q(q): break
    
    # Lower to spoon
    q_at_spoon = ik_pos_keepseed_L(spoon_center_local[0], spoon_center_local[1], pick_z, robot.q)
    for q in rtb.jtraj(robot.q, q_at_spoon, 20).q:
        if not step_q(q): break
    
    # GRAB SPOON: Calculate relative transform
    print("KUKA: Grabbing spoon...")
    Tee = robot.fkine(robot.q)
    T_spoon_rel = Tee.inv() * SE3(spoon_obj.T)
    grabbed = True
    update_spoon()
    
    # Lift spoon
    q_lift_spoon = ik_pos_keepseed_L(spoon_center_local[0], spoon_center_local[1], safe_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_lift_spoon, 20).q:
        if not step_q(q): break

    # Step 2 - Move to pot and stir
    print("KUKA: Moving to pot...")
    pre_potL = np.array([potL[0], potL[1], safe_zL])
    q_pre = ik_pos_keepseed_L(*pre_potL, robot.q)
    for q in rtb.jtraj(robot.q, q_pre, 40).q:
        if not step_q(q): break

    print("KUKA: Lowering into pot...")
    q_stir = ik_pos_keepseed_L(potL[0], potL[1], stir_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_stir, 20).q:
        if not step_q(q): break

    
    print("KUKA: Stirring with attached spoon...")
    R = float(stir_R)
    steps_per_rev = 24  
    seg_steps = 3       
    wrist_amp = np.deg2rad(wrist_sine_amp_deg)
    q_prev = robot.q.copy()

    for k in range(revs * steps_per_rev):
        if k % 5 == 0:
            SAFETY.wait_ok()
            if SAFETY.estop.is_set():
                print("E-STOP: KUKA stirring halted")
                break
                
        th = 2*np.pi*k/steps_per_rev
        xL = potL[0] + R*np.cos(th)
        yL = potL[1] + R*np.sin(th)
        
        
        q_tgt = ik_pos_smooth(xL, yL, stir_zL, q_prev)
        
        
        q_tgt[-1] += wrist_amp * np.sin(2*th)
        
        
        for qk in rtb.jtraj(q_prev, q_tgt, seg_steps).q:
            if not step_q(qk): break
            
         
        if k % 8 == 0:  
            env.add(Sphere(radius=0.008, pose=SE3(robot.fkine(robot.q).t), color=[0.0, 1.0, 0.0, 0.55]))
            
        q_prev = q_tgt.copy()

    # Step 4 - Lift out
    print("KUKA: Lifting out...")
    q_lift = ik_pos_keepseed_L(potL[0], potL[1], safe_zL, robot.q)
    for q in rtb.jtraj(robot.q, q_lift, 20).q:
        if not step_q(q): break

    print("KUKA stirring with pickup complete!")

# kuka function
def setup_kuka(env, which=1):
    """Creates KUKA robot and objects."""
    deg = pi / 180
    
    d1 = 0.400
    a_shoulder = 0.025
    a_upper    = 0.315
    a_forearm  = 0.365
    d4 = 0.035
    d6 = 0.080
    a2_merged = a_shoulder + a_upper

    A1 = (-2.9670597283903604,  2.9670597283903604)
    A2 = (-3.3161255787892263,  0.7853981633974483)
    A3 = (-2.0943951023931953,  2.722713633111154)
    A4 = (-3.2288591161895095,  3.2288591161895095)
    A5 = (-2.0943951023931953,  2.0943951023931953)
    A6 = (-6.1086523819801535,  6.1086523819801535)

    links = [
        rtb.RevoluteDH(a=0.0,       d=d1,        alpha=-pi/2, qlim=A1),
        rtb.RevoluteDH(a=a2_merged, d=0.0,       alpha= 0.0,  qlim=A2),
        rtb.RevoluteDH(a=a_forearm, d=0.0,       alpha=+pi/2, qlim=A3),
        rtb.RevoluteDH(a=0.0,       d=d4,        alpha=-pi/2, qlim=A4),
        rtb.RevoluteDH(a=0.0,       d=0.0,       alpha=+pi/2, qlim=A5),
        rtb.RevoluteDH(a=0.0,       d=d6,        alpha=+pi,   qlim=A6),
    ]
    kuka = rtb.DHRobot(links, name='KUKA KR 6 R700 sixx')
    kuka.q = np.array([0, -pi/2, 0, 0, 0, 0], dtype=float)

    robot_x = 0.25
    bench_x = robot_x - 1.2
    bench_y = (-0.8, 0.0, +0.8)[which]
    back_offset = -0.40
    
    # Keep the robot flipped 
    base_T = SE3(bench_x + back_offset, bench_y, 0.0) * SE3.Rz(np.pi)
    kuka.base = base_T

    cyl = CylindricalDHRobotPlot(kuka, cylinder_radius=0.04, multicolor=True)
    kuka_vis = cyl.create_cylinders()
    env.add(kuka_vis)

    
    tcp_home = kuka.fkine(kuka.q).t  
    
    spoon_center_local = np.array([-0.55, -0.20, 0.22])
    
    
    pot_center_local = np.array([-0.55, 0.00, 0.105])  
    
    pot_R = 0.11
    spoon_len = 0.18
    spoon_rad = 0.012

    spoon_pose_world = base_T * SE3(spoon_center_local)
    k_spoon = Cylinder(radius=spoon_rad, length=spoon_len, pose=spoon_pose_world, color=[0.95, 0.6, 0.2, 1.0])
    env.add(k_spoon)

    # Create bowl/pot at original position
    bowl_world_pose = base_T * SE3(pot_center_local)
    bowl_radius = pot_R * 0.9
    bowl_depth = 0.05

    k_bowl = Cylinder(radius=bowl_radius, length=bowl_depth, pose=bowl_world_pose, color=[0.8, 0.5, 0.2, 1.0])
    env.add(k_bowl)

    # Red rim markers around the pot
    for a in np.linspace(0, 2*pi, 36, endpoint=False):
        rim_local = pot_center_local + np.array([pot_R*np.cos(a), pot_R*np.sin(a), 0.0])
        env.add(Sphere(radius=0.01, pose=base_T * SE3(rim_local), color=[1.0, 0.0, 0.0, 0.7]))

    print(f"KUKA: Robot base at {base_T.t} (flipped 180°)")
    print(f"KUKA: Pot positioned at ORIGINAL local {pot_center_local} (world: {(base_T * SE3(pot_center_local)).t})")
    print(f"KUKA: TCP home position: {tcp_home}")
    
    return kuka, cyl, k_spoon, base_T, spoon_center_local, pot_center_local, pot_R, spoon_len


# Animation function for GP7
def animate_gp7_pick_and_stir(env, gp7, cyl, spoon_obj, base_T, pot_center_local,
                              safe_clear_z=0.05, wrist_wiggle_deg=20):
    """
    GP7 sequence: approach loose stick, attach (virtual grasp), move to pot, then stir.
    Uses the same jtraj/IK structure as your other robots and hands off to rmrc_stir_GP7 for stirring.
    """
    dt = 0.012

    
    grabbed = False
    T_spoon_rel = None
    def update_spoon():
        if grabbed and spoon_obj is not None and T_spoon_rel is not None:
            try:
                spoon_obj.T = (gp7.fkine(gp7.q) * T_spoon_rel).A
            except Exception:
                pass

    def min_link_height(q):
        T = np.eye(4); zmin = np.inf
        for i, L in enumerate(gp7.links):
            T = T @ L.A(q[i]).A
            zmin = min(zmin, T[2,3])
        return zmin

    def step_q(qnext):
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: GP7 animation halted"); return False
        if min_link_height(qnext) < (TABLE_TOP_Z + safe_clear_z):
            return False
        hits = collision_mgr.check(gp7, qnext)
        if hits:
            print(f"[LIVE] ⚠️ GP7 collision: {', '.join(hits)}"); SAFETY.estop_now(); return False
        gp7.q = qnext
        cyl.update(qnext)
        update_spoon()
        env.step(dt)
        return True

    def ik_pos(p, qseed=None):
        if qseed is None: qseed = gp7.q
        T = transl(p[0], p[1], p[2])
        sol = gp7.ikine_LM(T, q0=qseed, mask=[1,1,1,0,0,0])
        if sol.success: return sol.q
        # small vertical bias to help IK when near the table
        for dz in np.linspace(0.02, 0.12, 6):
            sol = gp7.ikine_LM(transl(p[0], p[1], p[2]+dz), q0=qseed, mask=[1,1,1,0,0,0])
            if sol.success: return sol.q
        return qseed

    def go(q_goal, n=100):
        for qk in rtb.jtraj(gp7.q, q_goal, n).q:
            if not step_q(qk): continue

    def go_linear_z(x, y, z_from, z_to, steps=70):
        qseed = gp7.q
        for z in np.linspace(z_from, z_to, steps):
            q = ik_pos(np.array([x,y,z]), qseed); qseed = q
            if not step_q(q): continue

    def go_linear_xy(xy_from, xy_to, z, steps=140):
        xs = np.linspace(xy_from[0], xy_to[0], steps)
        ys = np.linspace(xy_from[1], xy_to[1], steps)
        qseed = gp7.q
        for x,y in zip(xs, ys):
            q = ik_pos(np.array([x,y,z]), qseed); qseed = q
            if not step_q(q): continue

    # --- Where is the loose stick relative to the GP7 base? ---
    spoon_center_local = (base_T.inv() * SE3(spoon_obj.T)).t.reshape(-1)
    L = float(getattr(spoon_obj, "length", 0.18))

    # Heights
    z_clear = TABLE_TOP_Z + safe_clear_z
    safe_z  = max(spoon_center_local[2] + L/2, pot_center_local[2]) + 0.20

    # Waypoints
    pre_spoon = np.array([spoon_center_local[0], spoon_center_local[1], safe_z])
    top_spoon = spoon_center_local[2] + L/2.0 + 0.01
    pick_z    = max(top_spoon, z_clear + 0.02)
    pre_pot   = np.array([pot_center_local[0], pot_center_local[1], safe_z])
    stir_z    = pot_center_local[2] + 0.02

    # --- Sequence: approach -> descend -> attach -> lift -> move to pot -> descend
    q_pre_spoon = ik_pos(pre_spoon)
    go(q_pre_spoon, n=100)
    go_linear_z(spoon_center_local[0], spoon_center_local[1], safe_z, pick_z, steps=70)

    # Attach (virtual grasp): compute T_spoon_rel so the stick follows the TCP
    Tee = gp7.fkine(gp7.q)
    T_spoon_rel = Tee.inv() * SE3(spoon_obj.T)
    grabbed = True
    update_spoon()

    go_linear_z(spoon_center_local[0], spoon_center_local[1], pick_z, safe_z, steps=70)
    go_linear_xy(pre_spoon[:2], pre_pot[:2], safe_z, steps=140)
    q_pre_pot = ik_pos(pre_pot, gp7.q)
    go(q_pre_pot, n=60)
    go_linear_z(pot_center_local[0], pot_center_local[1], safe_z, stir_z, steps=70)

    # --- Hand off to your RMRC stir for GP7 (will keep the stick welded)
    rmrc_stir_GP7_smooth(env, gp7, cyl, spoon_obj, base_T, pot_center_local,
                  stir_R=0.085, stir_z_offset=0.02, seconds=20.0,
                  delta_t=0.05, revs=3, wrist_sine_amp_deg=wrist_wiggle_deg)

    # Lift out
    go_linear_z(pot_center_local[0], pot_center_local[1], stir_z, safe_z, steps=70)

# Kuka animation
def animate_kuka(env, kuka, cyl, k_spoon, base_T, spoon_center_local, pot_center_local, pot_R, spoon_len):
    """
    Runs the KUKA animation
    """
    print("Starting KUKA animation...")
    
    
    z_table = 0.00
    z_clear = z_table + 0.05
    safe_z  = max(spoon_center_local[2] + spoon_len/2, pot_center_local[2]) + 0.20
    kDT = 0.012

    grabbed = False
    T_spoon_rel = None

    def update_spoon():
        if grabbed:
            k_spoon.T = (kuka.fkine(kuka.q) * T_spoon_rel).A

    def min_link_height(q):
        T = np.eye(4)
        zmin = np.inf
        for i, L in enumerate(kuka.links):
            T = T @ L.A(q[i]).A
            zmin = min(zmin, T[2, 3])
        return zmin

    def step_q(qnext):
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: KUKA animation halted")
            return False
        if min_link_height(qnext) < z_clear:
            return False
        kuka.q = qnext
        cyl.update(qnext)
        update_spoon()
        env.step(kDT)
        return True

    def ik_pos(p, qseed=None):
        if qseed is None:
            qseed = kuka.q
        T = transl(p[0], p[1], p[2])  # in KUKA's base frame
        sol = kuka.ikine_LM(T, q0=qseed, mask=[1,1,1,0,0,0])
        if sol.success:
            return sol.q
        for dz in np.linspace(0.02, 0.12, 6):
            T = transl(p[0], p[1], p[2] + dz)
            sol = kuka.ikine_LM(T, q0=qseed, mask=[1,1,1,0,0,0])
            if sol.success:
                return sol.q
        return qseed

    def go(q_goal, n=120):
        qs = rtb.jtraj(kuka.q, q_goal, n).q
        for qk in qs:
            if not step_q(qk):
                continue

    def go_linear_z(x, y, z_from, z_to, steps=60):
        zs = np.linspace(z_from, z_to, steps)
        qseed = kuka.q
        for z in zs:
            q = ik_pos(np.array([x, y, z]), qseed)
            qseed = q
            if not step_q(q):
                continue

    def go_linear_xy(xy_from, xy_to, z, steps=120):
        xs = np.linspace(xy_from[0], xy_to[0], steps)
        ys = np.linspace(xy_from[1], xy_to[1], steps)
        qseed = kuka.q
        for x, y in zip(xs, ys):
            q = ik_pos(np.array([x, y, z]), qseed)
            qseed = q
            if not step_q(q):
                continue

    def mark():
        env.add(Sphere(radius=0.012, pose=SE3(kuka.fkine(kuka.q).t), color=[1.0, 0.0, 0.0, 0.55]))

    #
    pre_spoon = np.array([spoon_center_local[0], spoon_center_local[1], safe_z])
    top_spoon = spoon_center_local[2] + spoon_len/2
    pick_z    = max(top_spoon + 0.01, z_clear + 0.02)
    pre_pot   = np.array([pot_center_local[0], pot_center_local[1], safe_z])
    stir_z    = pot_center_local[2] + 0.02

    q_pre_spoon = ik_pos(pre_spoon)
    go(q_pre_spoon, n=100); mark()
    go_linear_z(spoon_center_local[0], spoon_center_local[1], safe_z, pick_z, steps=70); mark()

    Tee = kuka.fkine(kuka.q)
    T_spoon_rel = Tee.inv() * SE3(k_spoon.T)
    grabbed = True
    update_spoon()

    go_linear_z(spoon_center_local[0], spoon_center_local[1], pick_z, safe_z, steps=70); mark()
    go_linear_xy(pre_spoon[:2], pre_pot[:2], safe_z, steps=140); mark()
    q_pre_pot = ik_pos(pre_pot, kuka.q)
    go(q_pre_pot, n=60); mark()
    go_linear_z(pot_center_local[0], pot_center_local[1], safe_z, stir_z, steps=70); mark()

    #Stirring
    R = 0.085
    rev = 3
    steps_per_rev = 90
    seg_steps = 4
    wrist_amp = 20 * pi / 180

    def ik_pos_keepseed(x, y, z, qseed):
        T = transl(x, y, z)
        sol = kuka.ikine_LM(T, q0=qseed, mask=[1,1,1,0,0,0])
        return sol.q if sol.success else qseed

    q_prev = kuka.q.copy()
    N = rev * steps_per_rev
    g = 0
    for k in range(N):
        th = 2*pi*k/steps_per_rev
        x = pot_center_local[0] + R*np.cos(th)
        y = pot_center_local[1] + R*np.sin(th)
        q_target = ik_pos_keepseed(x, y, stir_z, q_prev)
        q_target[-1] = q_target[-1] + wrist_amp * np.sin(2*th)
        qs = rtb.jtraj(q_prev, q_target, seg_steps).q
        for qk in qs:
            if step_q(qk):
                if g % 3 == 0:
                    mark()
                g += 1
        q_prev = q_target.copy()

    go_linear_z(pot_center_local[0], pot_center_local[1], stir_z, safe_z, steps=70); mark()
    print("KUKA animation complete!")

# ABB animation
def animate_abb(env, abb, cyl, k_spoon, base_T, spoon_center, pot_center, pot_R, spoon_len):
    """
    ABB stirring motion — self-contained version of animate_kuka,
    cloned from the original ABB stick-moving demo.
    """
    print("Starting ABB stirring...")

    z_table = 0.00
    z_clear = z_table + 0.05
    safe_z  = max(spoon_center[2], pot_center[2]) + 0.20
    dt = 0.012

    def min_link_height(q):
        T = np.eye(4)
        zmin = np.inf
        for i, L in enumerate(abb.links):
            T = T @ L.A(q[i]).A
            zmin = min(zmin, T[2, 3])
        return zmin

    def step_q(qnext):
        SAFETY.wait_ok()
        if SAFETY.estop.is_set():
            print("E-STOP: ABB stirring halted")
            return False
        if min_link_height(qnext) < z_clear:
            return False
        abb.q = qnext
        cyl.update(qnext)
        env.step(dt)
        return True
   

    def ik_pos(p, qseed=None):
        if qseed is None:
            qseed = abb.q
        T = transl(p[0], p[1], p[2])
        sol = abb.ikine_LM(T, q0=qseed, mask=[1,1,1,0,0,0])
        if sol.success:
            return sol.q
        for dz in np.linspace(0.02, 0.12, 6):
            T = transl(p[0], p[1], p[2] + dz)
            sol = abb.ikine_LM(T, q0=qseed, mask=[1,1,1,0,0,0])
            if sol.success:
                return sol.q
        return qseed

    # approach spoon, pick up, move to pot, stir
    pre_spoon = np.array([spoon_center[0], spoon_center[1], safe_z])
    pick_z    = max(spoon_center[2] + 0.01, z_clear + 0.02)
    pre_pot   = np.array([pot_center[0], pot_center[1], safe_z])
    stir_z    = pot_center[2] + 0.02

    q_pre_spoon = ik_pos(pre_spoon)
    qs = rtb.jtraj(abb.q, q_pre_spoon, 100).q
    for q in qs: step_q(q)
    qs = rtb.jtraj(abb.q, ik_pos([spoon_center[0], spoon_center[1], pick_z]), 70).q
    for q in qs: step_q(q)
    qs = rtb.jtraj(abb.q, ik_pos([spoon_center[0], spoon_center[1], safe_z]), 70).q
    for q in qs: step_q(q)
    qs = rtb.jtraj(abb.q, ik_pos([pot_center[0], pot_center[1], safe_z]), 140).q
    for q in qs: step_q(q)

    q_pre_pot = ik_pos(pre_pot, abb.q)
    qs = rtb.jtraj(abb.q, q_pre_pot, 60).q
    for q in qs: step_q(q)
    qs = rtb.jtraj(abb.q, ik_pos([pot_center[0], pot_center[1], stir_z]), 70).q
    for q in qs: step_q(q)

    # Stir loop
    R = 0.085
    rev = 3
    steps_per_rev = 90
    seg_steps = 4
    wrist_amp = 20 * pi / 180
    q_prev = abb.q.copy()

    for k in range(rev * steps_per_rev):
        th = 2*pi*k/steps_per_rev
        x = pot_center[0] + R*np.cos(th)
        y = pot_center[1] + R*np.sin(th)
        q_target = ik_pos([x, y, stir_z], q_prev)
        q_target[-1] += wrist_amp * np.sin(2*th)
        qs = rtb.jtraj(q_prev, q_target, seg_steps).q
        for q in qs: step_q(q)
        q_prev = q_target.copy()

    qs = rtb.jtraj(abb.q, ik_pos([pot_center[0], pot_center[1], safe_z]), 70).q
    for q in qs: step_q(q)
    print("ABB stirring complete!")
# =============================================================================
# MAIN SCRIPT
# =============================================================================
env = swift.Swift()
env.launch(realtime=True)

start_hw_estop_bridge(port="COM3")   # <-- set your actual COM port

build_kitchen(env)
collision_mgr = CollisionManager(env)

# Front kitchen table (UR3)
collision_mgr.add_box("counter_kitchen_front",
                      center=[0.95, -0.15, 0.05],
                      half_extents=[0.60, 0.50, 0.05],
                      color=[0,1,0,0.2])

# Audience counters (GP7, KUKA, ABB)
audience_half = [0.35, 0.25, 0.05]
audience_centers = [
    [-0.95, -0.8, 0.05],  # GP7
    [-0.95,  0.0, 0.05],  # KUKA
    [-0.95,  0.8, 0.05],  # ABB
]
colors = [[1,0,0,0.2],[0,0,1,0.2],[1,1,0,0.2]]
for name, ctr, col in zip(["gp7_table","kuka_table","abb_table"],
                          audience_centers, colors):
    collision_mgr.add_box(name, ctr, audience_half, col)


# =============================================================================
# COMPACT SAFETY FENCE — shorter & thinner, enclosing full workspace
# =============================================================================
print("Adding compact safety fence around entire work area...")

# Lower height and thinner panels
fence_height = 0.55        # shorter barrier
panel_thickness = 0.05     # slimmer panels

# Left fence
collision_mgr.add_box("fence_left",
                      center=[-2.5, 0.0, fence_height/2],
                      half_extents=[panel_thickness, 2.3, fence_height/2],
                      color=[1.0, 0.5, 0.0, 0.55])

# Right fence
collision_mgr.add_box("fence_right",
                      center=[1.8, 0.0, fence_height/2],
                      half_extents=[panel_thickness, 2.3, fence_height/2],
                      color=[1.0, 0.5, 0.0, 0.55])

# Back fence
collision_mgr.add_box("fence_back",
                      center=[-0.35, -2.3, fence_height/2],
                      half_extents=[2.15, panel_thickness, fence_height/2],
                      color=[1.0, 0.5, 0.0, 0.55])

# Front fence
collision_mgr.add_box("fence_front",
                      center=[-0.35, 2.3, fence_height/2],
                      half_extents=[2.15, panel_thickness, fence_height/2],
                      color=[1.0, 0.5, 0.0, 0.55])

# Corner posts (a bit slimmer to match)
post_size = [0.08, 0.08, fence_height/2]
corners = [
    [-2.5, -2.3, fence_height/2],
    [-2.5,  2.3, fence_height/2],
    [ 1.8, -2.3, fence_height/2],
    [ 1.8,  2.3, fence_height/2],
]
for i, c in enumerate(corners, start=1):
    collision_mgr.add_box(f"fence_post_{i}",
                          center=c,
                          half_extents=post_size,
                          color=[0.6, 0.3, 0.0, 0.75])

print("Compact safety fence installed — lower and slimmer panels.")
# =============================================================================


env.add(Cuboid(scale=[0.5, 0.6, 0.6], pose=SE3(-0.35, -2.0, 0.3),
               color=[0.1, 0.1, 0.1, 1.0]))  # dark gray oven


env.add(Cuboid(scale=[0.5, 0.6, 0.6], pose=SE3(0.35, -2.0, 0.3),
               color=[0.75, 0.75, 0.8, 1.0]))  # metal sink plate

env.add(Cuboid(scale=[3.0, 3.0, 0.02], pose=SE3(-0.3, 0.0, 2.5),
               color=[1.0, 1.0, 0.9, 0.3])) # ceiling light

estop_center = SE3(-0.3, -2.94, 1.2)


# Red button dome (facing +Y into workspace)
env.add(Cylinder(radius=0.05, length=0.04,
                 pose=estop_center * SE3.Rx(pi/2) * SE3(0, 0.03, 0),
                 color=[1.0, 0.0, 0.0, 1.0]))       # red dome

# Metallic ring behind the button (outer trim)
env.add(Cylinder(radius=0.055, length=0.01,
                 pose=estop_center * SE3.Rx(pi/2) * SE3(0, 0.015, 0),
                 color=[0.8, 0.8, 0.8, 1.0]))       # silver rim

bin_height = 0.55
bin_radius = 0.18
bin_y = -2.75            # slightly in front of the back wall at y = -3.0
bin_z = bin_height / 2.0  # base sits on floor

# X positions for three bins (side-by-side)
bin_positions_x = [-0.7, -0.3, 0.1]

# Colors (grey = waste, blue = recycling, green = organics)
bin_colors = [
    [0.4, 0.4, 0.4, 1.0],   # grey
    [0.1, 0.2, 0.8, 1.0],   # blue
    [0.1, 0.5, 0.1, 1.0],   # green
]

# Add cylindrical bins
for x, color in zip(bin_positions_x, bin_colors):
    env.add(Cylinder(radius=bin_radius,
                     length=bin_height,
                     pose=SE3(x, bin_y, bin_z),
                     color=color))

# Add matching lids (flat discs)
for x, color in zip(bin_positions_x, bin_colors):
    lid_color = [min(1, c * 1.2) for c in color[:3]] + [1.0]
    env.add(Cylinder(radius=bin_radius, length=0.015,
                     pose=SE3(x, bin_y, bin_height),
                     color=lid_color))

# UR3
ur3 = make_ur3()
env.add(ur3)

# KUKA
kuka_data = setup_kuka(env, which=1)  # Returns all the KUKA components
kuka, cyl, k_spoon, base_T, spoon_center_local, pot_center_local, pot_R_kuka, spoon_len = kuka_data



# ABB robot setup
ABB_NEXT_SQUARE = SE3(-1.15, 1.0, 0.0)
abb2, abb2_cyl, abb2_vis = build_ABB_environment(env, base=ABB_NEXT_SQUARE, name="ABB2")

spoon_center_local2 = np.array([0.50, -0.35, 0.22])
pot_center_local2 = np.array([0.50, -0.20, 0.15])
pot_R2, spoon_len2, spoon_rad2 = 0.11, 0.18, 0.012

# ADD ABB spoon visual:
k_spoon2 = Cylinder(radius=spoon_rad2, length=spoon_len2,
                    pose=ABB_NEXT_SQUARE * SE3(spoon_center_local2),
                    color=[0.95, 0.6, 0.2, 1.0])
env.add(k_spoon2)

# ADD ABB pot visual:
abb_bowl_world_pose = ABB_NEXT_SQUARE * SE3(pot_center_local2)
abb_bowl_radius = pot_R2 * 0.9
abb_bowl_depth = 0.05

abb_bowl = Cylinder(radius=abb_bowl_radius, length=abb_bowl_depth, 
                    pose=abb_bowl_world_pose, color=[0.6, 0.4, 0.2, 1.0])
env.add(abb_bowl)

# ADD ABB pot rim markers (red):
for a in np.linspace(0, 2*np.pi, 36, endpoint=False):
    rim_local = pot_center_local2 + np.array([pot_R2*np.cos(a), pot_R2*np.sin(a), 0.0])
    env.add(Sphere(radius=0.01, pose=ABB_NEXT_SQUARE * SE3(rim_local), color=[1.0, 0.0, 0.0, 0.7]))

# GP7 robot setup 
GP7_BASE = SE3(-1.35, -0.75, 0.0)
gp7, gp7_cyl, gp7_stick = build_GP7_environment(env, base=GP7_BASE, name="GP7")

pot_center_local_gp7 = np.array([0.40, -0.075, 0.105])
pot_R_gp7 = 0.11

gp7_bowl_world_pose = GP7_BASE * SE3(pot_center_local_gp7)
gp7_bowl_radius = pot_R_gp7 * 0.9
gp7_bowl_depth = 0.05

gp7_bowl = Cylinder(radius=gp7_bowl_radius, length=gp7_bowl_depth, 
                    pose=gp7_bowl_world_pose, color=[0.6, 0.4, 0.2, 1.0])
env.add(gp7_bowl)

# GP7 pot rim markers 
for a in np.linspace(0, 2*np.pi, 36, endpoint=False):
    rim_local = pot_center_local_gp7 + np.array([pot_R_gp7*np.cos(a), pot_R_gp7*np.sin(a), 0.0])
    env.add(Sphere(radius=0.01, pose=GP7_BASE * SE3(rim_local), color=[0.0, 0.0, 1.0, 0.7]))
# Camera
try:
    env.set_camera_pose([2.4, 2.2, 1.7], [0.7, 0.05, 0.30])
except Exception:
    pass

# GUI: build shared sliders + robot selector 
try:
    robots = {}
    if 'ur3' in globals(): robots['ur3'] = ur3
    if 'kuka' in globals(): robots['kuka'] = kuka
    if 'abb2' in globals(): robots['abb'] = abb2
    if 'gp7' in globals(): robots['gp7'] = gp7
    if robots:
        build_gui_multi(env, robots, default_robot=('ur3' if 'ur3' in robots else next(iter(robots))))
        print("[GUI] Shared sliders + robot selector ready.")
    else:
        print("[GUI] No robots available for GUI.")
except Exception as _e:
    print("[GUI] Failed to start:", _e)
LC_POSE = SE3(-0.0, -0.10, 0.30)
#LC_POSE = SE3(0.45, -0.10, 0.30)
LC_SIZE = (0.02, 0.50, 0.40)
#LC_SIZE = (0.02, 0.50, 0.40)

LIGHT_CURTAIN = LightCurtain(
    pose=LC_POSE, size_xyz=LC_SIZE, env=env, name="LC",
    beam_rows=7, beam_cols=18, debounce_enter=0.0, debounce_clear=0.10
)
#LIGHT_CURTAIN = LightCurtain(
#    pose=LC_POSE, size_xyz=LC_SIZE, env=env, name="LC",
#    beam_rows=7, beam_cols=18, debounce_enter=0.05, debounce_clear=0.25
#)
LIGHT_CURTAIN.add_to_env()
LIGHT_CURTAIN.start_monitor(robots=robots, status_label=None, poll = 0.01)    

#ur3 poses
q_home        = np.array([0.0,  -np.pi/3, np.pi/3, -np.pi/2, -np.pi/2, 0.0])
q_above_spoon = np.array([-0.5, -np.pi/4, np.pi/4, -np.pi/2, -np.pi/2, 0.0])
q_at_spoon    = np.array([-0.5, -np.pi/6, np.pi/6, -np.pi/2, -np.pi/2, 0.0])
q_lift_spoon  = np.array([-0.5, -np.pi/3, np.pi/3, -np.pi/2, -np.pi/2, 0.0])
q_above_pot   = np.array([ 0.3, -np.pi/4, np.pi/4, -np.pi/2, -np.pi/2, 0.0])
q_in_pot      = np.array([ 0.3, -np.pi/6, np.pi/6, -np.pi/2, -np.pi/2, 0.0])


p_spoon_tcp = np.asarray(ur3.fkine(q_at_spoon).t).reshape(-1)
SPOON_CENTER_UR3 = np.array([p_spoon_tcp[0], p_spoon_tcp[1], p_spoon_tcp[2] - SPOON_LEN/2.0])

p_pot_tcp = np.asarray(ur3.fkine(q_in_pot).t).reshape(-1)
POT_CENTER_UR3 = np.array([p_pot_tcp[0], p_pot_tcp[1], p_pot_tcp[2]])

#pots and spoons
spoon_ur3 = Cylinder(radius=SPOON_RAD, length=SPOON_LEN, pose=SE3(*SPOON_CENTER_UR3), color=[0.95,0.6,0.2,1])
env.add(spoon_ur3)
for a in np.linspace(0, 2*np.pi, 36, endpoint=False):
    rim = POT_CENTER_UR3 + np.array([POT_R*np.cos(a), POT_R*np.sin(a), 0.0])
    env.add(Sphere(radius=0.01, pose=SE3(*rim), color=[0.85, 0.10, 0.10, 1]))

print("Starting UR3 cooking sequence...")

##UR3 Cooking Sequence
print("1. UR3 moving above spoon...")
traj1 = rtb.jtraj(ur3.q, q_above_spoon, STEPS_MED).q
ur3.q = safe_execute_trajectory(ur3, traj1)

print("2. UR3 descending to spoon...")
traj2 = rtb.jtraj(ur3.q, q_at_spoon, STEPS_FAST).q
ur3.q = safe_execute_trajectory(ur3, traj2)

print("3. UR3 attaching spoon...")
Tee_current = ur3.fkine(ur3.q)
T_spoon_current = SE3(spoon_ur3.T)
T_spoon_rel = Tee_current.inv() * T_spoon_current
attached = True
time.sleep(0.5)

print("4. UR3 lifting spoon...")
traj3 = rtb.jtraj(ur3.q, q_lift_spoon, STEPS_MED).q
ur3.q = safe_execute_trajectory(ur3, traj3, spoon_attached=attached, spoon_obj=spoon_ur3, T_rel=T_spoon_rel)

print("5. UR3 moving to pot...")
traj4 = rtb.jtraj(ur3.q, q_above_pot, STEPS_SLOW).q
ur3.q = safe_execute_trajectory(ur3, traj4, spoon_attached=attached, spoon_obj=spoon_ur3, T_rel=T_spoon_rel)

print("6. UR3 lowering into pot...")
traj5 = rtb.jtraj(ur3.q, q_in_pot, STEPS_FAST).q
ur3.q = safe_execute_trajectory(ur3, traj5, spoon_attached=attached, spoon_obj=spoon_ur3, T_rel=T_spoon_rel)

print("7. UR3 stirring...")
base_config = ur3.q.copy()
stir_radius = 0.05
stir_speed  = 0.06
num_circles = 3
for i in range(int(num_circles * 2 * np.pi / stir_speed)):
    SAFETY.wait_ok()
    if SAFETY.estop.is_set():
        print("E-STOP: UR3 stirring halted")
        break
    angle = i * stir_speed
    q_stir = base_config.copy()
    q_stir[0] = base_config[0] + stir_radius * np.cos(angle)
    q_stir[1] = base_config[1] + stir_radius * 0.5 * np.sin(angle)
    ur3.q = q_stir
    try:
        Tee = ur3.fkine(q_stir)
        spoon_ur3.T = (Tee * T_spoon_rel).A
    except Exception:
        pass
    env.step(DT)

print("8. UR3 lifting out...")
traj_final = rtb.jtraj(ur3.q, q_above_pot, STEPS_MED).q
ur3.q = safe_execute_trajectory(ur3, traj_final, spoon_attached=attached, spoon_obj=spoon_ur3, T_rel=T_spoon_rel)

print("UR3 cooking sequence complete!")
print("Starting ABB RMRC stirring...")
print("Starting ABB RMRC stirring...")

# Executes RMRC stirring
rmrc_stir_ABB_with_pickup(
    env,
    abb2,
    abb2_cyl, 
    k_spoon2, 
    ABB_NEXT_SQUARE,
    pot_center_local2,
    stir_R=0.085, 
    tip_depth=0.02,  
    seconds=20.0,
    delta_t = 0.05, 
    revs=3, 
    wrist_sine_amp_deg=20,  
    max_joint_vel=1.5       
)
print("ABB RMRC stirring complete!")

print("KUKA Smooth RMRC stirring...")
rmrc_stir_KUKA_with_pickup(
    env, kuka, cyl, k_spoon, base_T, pot_center_local,
    stir_R=0.085, 
    tip_depth=0.02,         
    seconds=15.0, 
    delta_t=0.04,
    revs=3, 
    wrist_sine_amp_deg=15, 
    max_joint_vel=1.2
)
print("KUKA Smooth RMRC stirring complete!")

print("GP7 stirring...")
rmrc_stir_GP7_smooth(
    env,
    gp7,                    
    gp7_cyl,                
    gp7_stick,              
    GP7_BASE,               
    pot_center_local_gp7,   
    stir_R=0.06,
    tip_depth=0.02,         
    seconds=20.0,
    delta_t=0.05,
    revs=3,
    wrist_sine_amp_deg=20,
    max_joint_vel=1.5       
)
print("GP7 stirring complete!")
env.hold()

