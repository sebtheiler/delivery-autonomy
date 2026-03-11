from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion

def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    qx, qy, qz, qw = rot.as_quat()
    
    return Quaternion(x=qx, y=qy, z=qz, w=qw)
