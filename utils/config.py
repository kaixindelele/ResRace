import math
import numpy

Config = {
    'lidar_type' :      'Velodyne_16',
    'frequency' :       float(20),
    'lines' :           int(675),
    'range' :           float(10.0),
    'angle_lower' :     float(-0.75 * math.pi),
    'angle_upper' :     float(+0.75 * math.pi),
    'sys_error' :       float(0.03),
    'angle_acc' :       float(0.4*math.pi/180),
    'imu_dim' :         int(8),
    'lidar_dim' :       int(675),
    'action_dim':       int(2),
    'safe_margin' :     float(0.25),
    'demo_epoch':       int(5),
    'agent_p_thres':    float(0.8),
}

APF_Config = {
    'austria_wide':     float(2.0),
    'barcelona':        float(2.0),
    'columbia':         float(2.0),
    'montreal':         float(2.0),
    'plechaty':         float(2.0),
    
    'gbr':              float(2.0),
    'circle_cw':        float(0.5),
}
