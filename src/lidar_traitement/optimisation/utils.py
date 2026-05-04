from dataclasses import dataclass, field, astuple
from typing import List, Optional, Tuple

import numpy as np
import sys
import math

from sklearn import covariance


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Identique à l'original — déjà bien écrit."""
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    return (s0 * q0 + s1 * q1) / np.linalg.norm(s0 * q0 + s1 * q1)

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

    def to_list(self):
        return [self.x, self.y, self.z]

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
    position: "Position" = field(default_factory=Position)
    orientation: "Quaternion" = field(default_factory=Quaternion)
    covariance: list[float] = field(default_factory=lambda: [0.0] * 36)

    def printData(self, decalage: str = ""):
        print(f"{decalage}=== PoseWithCovariance ===")
        self.position.printData(decalage + "  ")
        self.orientation.printData(decalage + "  ")
        print(f"{decalage}covariance : {self.covariance}")

    @staticmethod
    def _data_conv_to_array(data: "PoseWithCovariance") -> np.ndarray:
        if data.position is None or data.orientation is None:
            raise ValueError("position et orientation ne doivent pas être None")

        return np.array([
            [data.position.x, data.position.y, data.position.z, 0.0],
            [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        ], dtype=np.float32)

    @staticmethod
    def _array_conv_to_data(data: np.ndarray) -> "PoseWithCovariance":
        if data.shape != (2, 4):
            raise ValueError("Le tableau doit être de forme (2, 4)")

        return PoseWithCovariance(
            position=Position(data[0, 0], data[0, 1], data[0, 2]),
            orientation=Quaternion(data[1, 0], data[1, 1], data[1, 2], data[1, 3]),
            covariance=[0.0] * 36
        )


@dataclass
class TwistWithCovariance:
    linear: Optional["Vector3D"] = field(default_factory=Vector3D)
    angular: Optional["Vector3D"] = field(default_factory=Vector3D)
    covariance: List[float] = field(default_factory=lambda: [0.0] * 36)

    def printData(self, decalage: str = ""):
        print(f"{decalage}=== TwistWithCovariance ===")

        if self.linear is not None:
            self.linear.printData(decalage + "  ")
        else:
            print(f"{decalage}  linear: None")

        if self.angular is not None:
            self.angular.printData(decalage + "  ")
        else:
            print(f"{decalage}  angular: None")

        print(f"{decalage}covariance : {self.covariance}")

    @staticmethod
    def _data_conv_to_array(data: "TwistWithCovariance") -> np.ndarray:
        if data.linear is None or data.angular is None:
            raise ValueError("linear et angular ne doivent pas être None")

        return np.array([
            [data.linear.x, data.linear.y, data.linear.z],
            [data.angular.x, data.angular.y, data.angular.z]
        ],dtype=np.float32)

    @staticmethod
    def _array_conv_to_data(data: np.ndarray) -> "TwistWithCovariance":
        if data.shape != (2, 3):
            raise ValueError("Le tableau doit être de forme (2, 3)")

        return TwistWithCovariance(
            linear=Vector3D(data[0, 0], data[0, 1], data[0, 2]),
            angular=Vector3D(data[1, 0], data[1, 1], data[1, 2]),
            covariance=[0.0] * 36
        )


@dataclass
class Odometry:
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)
    current_yaw: float = 0.0

    def printData(self,decalage=""):
        print(f"{decalage}=== Odometry ===")
        #self.header.printData(decalage + "  ")
        print(f"{decalage}   child_frame_id : "+self.child_frame_id)
        self.pose.printData(decalage + "  ")
        self.twist.printData(decalage + "  ")


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
    """
    OPTIMISÉ :
    - Quaternions stockés en numpy array (4, N) → accès O(1) sans copie
    - searchsorted déjà utilisé correctement, conservé
    - _quat_to_yaw vectorisée pour usage batch possible
    """

    def __init__(self, gnss_msgs):
        import sys
        if not gnss_msgs:
            sys.exit("[ERREUR] Aucun message GNSS dans le bag.")

        n = len(gnss_msgs)
        self.timestamps  = np.empty(n, dtype=np.float64)
        self.positions   = np.empty((n, 3), dtype=np.float64)
        # OPTIMISATION : stockage vectorisé [w, x, y, z]
        self.quaternions = np.empty((n, 4), dtype=np.float64)
        self._gnss_speeds = np.empty(n, dtype=np.float64)

        for i, (ts, msg) in enumerate(gnss_msgs):
            try:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
            except AttributeError:
                sys.exit("[ERREUR] Topic GNSS attendu : PoseWithCovarianceStamped")
            self.timestamps[i]   = ts * 1e-9
            self.positions[i]    = (p.x, p.y, p.z)
            self.quaternions[i]  = (q.w, q.x, q.y, q.z)  # [w,x,y,z]

        self._compute_gnss_speeds()
        print(f"[INFO] Trajectoire GNSS : {n} points.")

    def _compute_gnss_speeds(self):
        dt  = np.diff(self.timestamps)
        dp  = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        spd = np.divide(dp, dt, out=np.zeros_like(dp), where=dt > 1e-6)
        self._gnss_speeds[0]  = 0.0
        self._gnss_speeds[1:] = spd

    def interpolate_pose_se3(self, query_ts_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne (position_xyz, quaternion_wxyz) interpolés."""
        t = query_ts_ns * 1e-9
        ts = self.timestamps

        if t < ts[0] or t > ts[-1]:
            return None, None

        idx = np.searchsorted(ts, t)
        if idx == 0:
            return self.positions[0].copy(), self.quaternions[0].copy()
        if idx >= len(ts):
            return self.positions[-1].copy(), self.quaternions[-1].copy()

        alpha = (t - ts[idx - 1]) / (ts[idx] - ts[idx - 1] + 1e-12)
        pos = self.positions[idx - 1] + alpha * (self.positions[idx] - self.positions[idx - 1])
        # OPTIMISATION : accès direct numpy, pas de _quat_to_array()
        q_interp = slerp(self.quaternions[idx - 1], self.quaternions[idx], alpha)
        return pos, q_interp

    def interpolate_pose(self, query_ts_ns: int):
        """Rétro-compatibilité : retourne (pos, yaw)."""
        pos, q = self.interpolate_pose_se3(query_ts_ns)
        if q is None:
            return None, None
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        return pos, yaw

    def gnss_speed_at(self, query_ts_ns: int) -> float:
        t   = query_ts_ns * 1e-9
        idx = min(np.searchsorted(self.timestamps, t), len(self._gnss_speeds) - 1)
        return float(self._gnss_speeds[idx])