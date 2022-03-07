import math
import random
import numpy as np
from utils.config import Config
'''
Config = {
    'lidar_type' :      'Velodyne_16',
    'frequency' :       float(20),
    'lines' :           int(675),
    'range' :           float(10.0),
    'angle_lower' :     float(-0.75 * math.pi),
    'angle_upper' :     float(+0.75 * math.pi),
    'sys_error' :       float(0.03),
    'angle_acc' :       float(0.4*math.pi/180),
    'imu_dim' :         int(7),
    'lidar_dim' :       int(675),
    'safe_margin' :     float(0.42),
}
'''
def Generate_Goal(lidar_obs, uncon_thres=1.5):
    '''
    Generate the sub-goal in a moveable window for lidar observations.
    2 Parts: The most remote point / The edge point
    '''
    goal_rau, goal_theta = 0, 0
    lidar_list = lidar_obs.tolist()
    '''
    for i in range(562, 113, -1):

        i_previous = sum(lidar_list[i-5:i])/5
        i_tail = sum(lidar_list[i:i+5])/5

        if abs(i_previous - i_tail) >= uncon_thres:
            goal_theta = Config['angle_upper'] - i*Config['angle_acc']
            goal_rau = 0.5*(i_previous + i_tail)
            break
        elif 0.5*(i_previous+i_tail) >= Config['range'] - 1.0:
            goal_theta = Config['angle_upper'] - i*Config['angle_acc']
            goal_rau = Config['range'] - 1.0
            break
        else:
            id_max = np.argmax(lidar_obs)
            goal_theta = Config['angle_upper'] - id_max*Config['angle_acc']
            goal_rau = lidar_obs[id_max]
    '''
    
    for i in range(342, 562):

        i_previous = sum(lidar_list[i-5:i])/5
        i_tail = sum(lidar_list[i:i+5])/5
        i_previous_2 = sum(lidar_list[(674-i-5):(674-i)])/5
        i_tail_2 = sum(lidar_list[(674-i):(674-i+5)])/5

        if abs(i_previous - i_tail) >= uncon_thres:
            goal_theta = Config['angle_upper'] - i*Config['angle_acc']
            goal_rau = 0.5*(i_previous + i_tail)
            break
        elif abs(i_previous_2 - i_tail_2) >= uncon_thres:
            goal_theta = Config['angle_upper'] - (674-i)*Config['angle_acc']
            goal_rau = 0.5*(i_previous_2 + i_tail_2)
            break
        elif 0.5*(i_previous+i_tail) >= Config['range'] - 1.0:
            goal_theta = Config['angle_upper'] - i*Config['angle_acc']
            goal_rau = Config['range'] - 1.0
            break
        elif 0.5*(i_previous_2+i_tail_2) >= Config['range'] - 1.0:
            goal_theta = Config['angle_upper'] - (674-i)*Config['angle_acc']
            goal_rau = Config['range'] - 1.0
            break
        else:
            id_max = np.argmax(lidar_obs)
            goal_theta = Config['angle_upper'] - id_max*Config['angle_acc']
            goal_rau = lidar_obs[id_max]
    '''
    for i in range(548, 890):

        i_previous = sum(lidar_list[i-8:i])/8
        i_tail = sum(lidar_list[i:i+8])/8
        i_previous_2 = sum(lidar_list[(1079-i-8):(1079-i)])/8
        i_tail_2 = sum(lidar_list[(1079-i):(1079-i+8)])/8

        if abs(i_previous - i_tail) >= uncon_thres:
            goal_theta = Config['angle_upper'] - i*Config['angle_acc']
            goal_rau = 0.5*(i_previous + i_tail)
            break
        elif abs(i_previous_2 - i_tail_2) >= uncon_thres:
            goal_theta = Config['angle_upper'] - (1079-i)*Config['angle_acc']
            goal_rau = 0.5*(i_previous_2 + i_tail_2)
            break
        elif 0.5*(i_previous+i_tail) >= Config['range'] - 1.0:
            goal_theta = Config['angle_upper'] - i*Config['angle_acc']
            goal_rau = Config['range'] - 1.0
            break
        elif 0.5*(i_previous_2+i_tail_2) >= Config['range'] - 1.0:
            goal_theta = Config['angle_upper'] - (1079-i)*Config['angle_acc']
            goal_rau = Config['range'] - 1.0
            break
        else:
            id_max = np.argmax(lidar_obs)
            goal_theta = Config['angle_upper'] - id_max*Config['angle_acc']
            goal_rau = lidar_obs[id_max]
    '''
    goal_x, goal_y = goal_rau*math.cos(goal_theta), goal_rau*math.sin(goal_theta)
    return goal_x, goal_y



def Modified_APF(lidar_obs, eta_att=1, eta_rep_x=0.075, eta_rep_y=0.1, rep_range=6*Config['safe_margin'], ita=0.1):
    '''
    Calculate Repulsive Force
    '''
    force_rx, force_ry = 0, 0
    for i in range(Config['lines']):
        line_angle = Config['angle_upper'] - i*Config['angle_acc']
        line_length = lidar_obs[i]
        rep_force = (1/line_length+1/rep_range)/line_length+0.5*(1/line_length+1/rep_range)**2
        if lidar_obs[i] < rep_range:
            force_rx_i = -eta_rep_x*rep_force*math.cos(line_angle)
            force_ry_i = -eta_rep_y*rep_force*math.sin(line_angle)
        else:
            force_rx_i, force_ry_i = 0, 0
        force_rx += force_rx_i
        force_ry += force_ry_i

    '''
    Calculate Attractive Force
    '''
    goal_x, goal_y = Generate_Goal(lidar_obs)
    goal_dis = math.sqrt(goal_x**2 + goal_y**2)
    force_ax = eta_att*goal_x*goal_dis
    force_ay = eta_att*goal_y*goal_dis
    force_x, force_y = (force_rx+force_ax), (force_ry+force_ay)
    random_rate = random.uniform(0, 1)
    if force_rx < 0 and abs(goal_x/goal_y)<3**(-0.5) and random_rate<=ita:
        force_x = 0.1
    return force_x, force_y

def Nonlinear_Controller(lidar_obs, rep_range=6*Config['safe_margin']):
    force_x, force_y = Modified_APF(lidar_obs, rep_range=rep_range)
    steering = -1 * math.tanh(force_y/50) + random.gauss(0, 0.01)
    motor = math.tanh(force_x/20) + random.gauss(0, 0.01)
    return np.clip(np.array([steering, motor]), -1, +1)




    
        

    
