from optimisation.utils import Odometry, FourWheelSteeringStamped
import numpy as np


# def FWSS_by_Odometry(fwss : FourWheelSteeringStamped, pred : Odometry) -> Odometry:
#     angular = np.abs(fwss.front_steering_angle-fwss.rear_steering_angle)/2
#     angular = angular + pred.pose.orientation.z
#
#     odom = Odometry()
#     odom.twist.linear.x = np.cos(angular) * fwss.speed
#     odom.twist.linear.y = np.sin(angular) * fwss.speed
#     odom.twist.angular.z = angular
#
#     odom.pose.position.x = pred.pose.position.x + odom.twist.linear.x
#     odom.pose.position.y = pred.pose.position.y + odom.twist.linear.y
#     odom.pose.orientation.z = angular
#     print(f"coord x : {odom.pose.position.x}")
#     print(f"coord y : {odom.pose.position.y}")
#     print(f"angle : {angular}")
#
#     return odom

def FWSS_by_Odometry(fwss : FourWheelSteeringStamped, delta_t: tuple[float, float], pred : Odometry,  L : float = 2.5) -> Odometry:
    delta_ = (delta_t[0] - delta_t[1]) / 1_000_000_000.0

    odom = pred
    angle_f = fwss.front_steering_angle
    angle_r = fwss.rear_steering_angle

    # 1. Angle de dérive
    beta = np.arctan((np.tan(angle_f) + np.tan(angle_r)) / 2.0)

    # 2. Vitesse de rotation (Yaw Rate)
    # Note le signe "-" : pour tourner, l'arrière doit être opposé à l'avant
    odom.twist.angular.z = (fwss.speed * np.cos(beta) / L) * (np.tan(angle_f) - np.tan(angle_r))

    # 3. Mise à jour de l'orientation (Yaw)
    if not hasattr(pred, 'current_yaw'):
        pred.current_yaw = 0.0

    odom.pose.angular.z += odom.twist.angular.z * delta_

    # 4. Vitesses globales

    odom.twist.linear.x = fwss.speed * np.cos(pred.current_yaw + beta)
    odom.twist.linear.y = fwss.speed * np.sin(pred.current_yaw + beta)

    # 5. On bouge !
    odom.deplacement(delta_)
    return odom


def get_transformation_matrix(fwss, delta_t, L=2.5):
    # 1. Calculs cinématiques de base (déjà faits)
    delta_t_sec = delta_t / 1_000_000_000.0

    tan_sum = np.tan(fwss.front_steering_angle) + np.tan(fwss.rear_steering_angle)
    tan_diff = np.tan(fwss.front_steering_angle) - np.tan(fwss.rear_steering_angle)

    beta = np.arctan(tan_sum / 2.0)
    yaw_rate = (fwss.speed * np.cos(beta) / L) * tan_diff

    # 2. Mouvement relatif (local au véhicule)
    d_psi = yaw_rate * delta_t_sec
    dx = fwss.speed * np.cos(beta) * delta_t_sec
    dy = fwss.speed * np.sin(beta) * delta_t_sec

    # 3. Construction de la matrice 3x3
    T = np.array([
        [np.cos(d_psi), -np.sin(d_psi), dx],
        [np.sin(d_psi), np.cos(d_psi), dy],
        [0, 0, 1]
    ])

    return T