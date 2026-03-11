from seblib.bicycle_model import bicycle_model

STATE_DIMENSION = 4
INPUT_DIMENSION = 2
WHEELBASE = 0.5 # robot.urdf.xacro

def dynamics_model(x, u, dt):
    return bicycle_model(x, u, dt, WHEELBASE)
