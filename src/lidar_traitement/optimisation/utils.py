from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import sys
import math



def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP entre deux quaternions [w,x,y,z].
    Gère : antipodalité, quaternions quasi-identiques, t ∈ [0,1].
    """
    # Assure le chemin court (hemisphere positif)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)

    if dot > 0.9995:                        # quasi-identiques → lerp linéaire
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_0   = math.sin(theta_0)

    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    q  = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)

@dataclass
class Header:
    frame_id: str = ""
    sec: int = 0
    nanosec: int = 0
    def printData(self,decalage=""):
        print(f"{decalage}=== Header ===")
        print(f"{decalage}   frame_id : "+self.frame_id)
        print(f"{decalage}   sec : {self.sec}")
        print(f"{decalage}   nanosec : {self.nanosec}")

@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    def printData(self,decalage=""):
        print(f"{decalage}=== Vector3D ===")
        print(f"{decalage}x : {self.x}")
        print(f"{decalage}y : {self.y}")
        print(f"{decalage}z : {self.z}")

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    def printData(self,decalage=""):
        print(f"{decalage}=== Position ===")
        print(f"{decalage}x : {self.x}")
        print(f"{decalage}y : {self.y}")
        print(f"{decalage}z : {self.z}")

@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def printData(self,decalage=""):
        print(f"{decalage}=== Quaternion ===")
        print(f"{decalage}   x : {self.x}")
        print(f"{decalage}   y : {self.y}")
        print(f"{decalage}   z : {self.z}")
        print(f"{decalage}   w : {self.w}")


@dataclass
class PoseWithCovariance:
    position: Position = Position
    orientation: Quaternion = Quaternion
    covariance: list[float] = field(default_factory=lambda: [0.0] * 36)

    def printData(self,decalage=""):
        print(f"{decalage}=== PoseWithCovariance ===")
        self.position.printData(decalage)
        self.orientation.printData(decalage)
        print(f"{decalage}covariance : {self.covariance}")



@dataclass
class TwistWithCovariance:
    linear: Optional[Vector3D] = Vector3D
    angular: Optional[Vector3D] = Vector3D
    covariance: List[float] = field(default_factory=lambda: [0.0] * 36)

    def printData(self,decalage=""):
        print(f"{decalage}=== FourWheelSteeringStamped ===")
        self.linear.printData(decalage + "  ")
        self.angular.printData(decalage + "  ")
        print(f"{decalage}covariance : {self.covariance}")


@dataclass
class Odometry:
    header: Optional[Header] = Header
    child_frame_id: str = ""
    pose: Optional[PoseWithCovariance] = PoseWithCovariance
    twist: Optional[TwistWithCovariance] = TwistWithCovariance

    def printData(self,decalage=""):
        print(f"{decalage}=== Odometry ===")
        #self.header.printData(decalage + "  ")
        print(f"{decalage}   child_frame_id : "+self.child_frame_id)
        self.pose.printData(decalage + "  ")
        self.twist.printData(decalage + "  ")
    def deplacement(self, delta):
        self.pose.position.x = self.pose.position.x + self.twist.linear.x * delta
        self.pose.position.y = self.pose.position.y + self.twist.linear.y * delta


@dataclass
class FourWheelSteeringStamped:
    front_steering_angle: float = 0.0
    rear_steering_angle: float = 0.0
    front_steering_angle_velocity: float = 0.0
    rear_steering_angle_velocity: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0

    def _conver_MSG(self, msg):
        self.front_steering_angle = msg.data.front_steering_angle
        self.rear_steering_angle = msg.data.rear_steering_angle
        self.front_steering_angle_velocity = msg.data.front_steering_angle_velocity
        self.rear_steering_angle_velocity = msg.data.rear_steering_angle_velocity
        self.speed = msg.data.speed
        self.acceleration = msg.data.acceleration
        self.jerk = msg.data.jerk

    def printData(self,decalage=""):
        print(f"{decalage}=== FourWheelSteeringStamped ===")
        print(f"{decalage}   front_steering_angle : {self.front_steering_angle}")
        print(f"{decalage}   rear_steering_angle : {self.rear_steering_angle}")
        print(f"{decalage}   front_steering_angle_velocity : {self.front_steering_angle_velocity}")
        print(f"{decalage}   rear_steering_angle_velocity : {self.rear_steering_angle_velocity}")
        print(f"{decalage}   speed : {self.speed}")
        print(f"{decalage}   acceleration : {self.acceleration}")
        print(f"{decalage}   jerk : {self.jerk}")

class GNSSTrajectory:
    """Position + orientation interpolées depuis PoseWithCovarianceStamped."""

    def __init__(self, gnss_msgs):
        if not gnss_msgs:
            sys.exit("[ERREUR] Aucun message GNSS dans le bag.")

        n = len(gnss_msgs)
        self.timestamps = np.empty(n, dtype=np.float64)
        self.positions  = np.empty((n, 3), dtype=np.float64)
        self.quaternions = []
        self._gnss_speeds = np.empty(n, dtype=np.float64)

        for i, (ts, msg) in enumerate(gnss_msgs):
            try:
                p = msg.pose.pose.position
                self.quaternions.append(msg.pose.pose.orientation)
            except AttributeError:
                sys.exit("[ERREUR] Topic GNSS attendu : PoseWithCovarianceStamped")
            self.timestamps[i]  = ts * 1e-9
            self.positions[i]   = (p.x, p.y, p.z)

        self._compute_gnss_speeds()
        print(f"[INFO] Trajectoire GNSS : {n} points.")

    def _compute_gnss_speeds(self):
        # Vectorisé au lieu d'une boucle Python
        dt  = np.diff(self.timestamps)
        dp  = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        spd = np.divide(dp, dt, out=np.zeros_like(dp), where=dt > 1e-6)
        self._gnss_speeds[0]  = 0.0
        self._gnss_speeds[1:] = spd

    def interpolate_pose(self, query_ts_ns: int):
        t  = query_ts_ns * 1e-9
        ts = self.timestamps

        if t < ts[0] or t > ts[-1]:
            return None, None

        idx = np.searchsorted(ts, t)
        if idx == 0:
            return self.positions[0].copy(), self._quat_to_yaw(self.quaternions[0])
        if idx >= len(ts):
            return self.positions[-1].copy(), self._quat_to_yaw(self.quaternions[-1])

        alpha = (t - ts[idx - 1]) / (ts[idx] - ts[idx - 1] + 1e-12)
        pos   = self.positions[idx - 1] + alpha * (self.positions[idx] - self.positions[idx - 1])
        q     = self.quaternions[idx - 1] if alpha < 0.5 else self.quaternions[idx]
        return pos, self._quat_to_yaw(q)

    def interpolate_pose_se3(self, query_ts_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne (position_xyz, quaternion_wxyz) interpolés."""
        t = query_ts_ns * 1e-9
        ts = self.timestamps

        if t < ts[0] or t > ts[-1]:
            return None, None

        idx = np.searchsorted(ts, t)
        if idx == 0:
            return self.positions[0].copy(), self._quat_to_array(self.quaternions[0])
        if idx >= len(ts):
            return self.positions[-1].copy(), self._quat_to_array(self.quaternions[-1])

        alpha = (t - ts[idx - 1]) / (ts[idx] - ts[idx - 1] + 1e-12)

        # LERP linéaire pour la position
        pos = self.positions[idx - 1] + alpha * (self.positions[idx] - self.positions[idx - 1])

        # SLERP pour le quaternion
        q0 = self._quat_to_array(self.quaternions[idx - 1])
        q1 = self._quat_to_array(self.quaternions[idx])
        q_interp = slerp(q0, q1, alpha)

        return pos, q_interp

    @staticmethod
    def _quat_to_array(q):
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)

    def gnss_speed_at(self, query_ts_ns: int) -> float:
        t   = query_ts_ns * 1e-9
        idx = min(np.searchsorted(self.timestamps, t),
                  len(self._gnss_speeds) - 1)
        return float(self._gnss_speeds[idx])

    @staticmethod
    def _quat_to_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))