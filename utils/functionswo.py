import math
import random
import numpy as np
import keyboard
import sys
import tty
import termios
import time
'''
def lidar2binary(lidar_obs, map_size=[250, 250]):
    p_augment = random.uniform(0, 1)

    binary_map = np.ones(map_size)
    center = [125, 125]
    binary_map[center[0]-5:center[0]+6, center[1]-6:center[1]+7] = 0.67
    counter = 0
    for i in range(675):
        if lidar_obs[i] > 5.0:
            continue
        else:
            angle = math.pi*(135+0.4*i)/180
            point_loc_x = int(center[0]+math.cos(angle)*round(lidar_obs[i], 3)*25)
            point_loc_y = int(center[1]-math.sin(angle)*round(lidar_obs[i], 3)*25)
            binary_map[point_loc_x-2:point_loc_x+3, point_loc_y-2:point_loc_y+3] = 0.33
    if p_augment > 0.667:
        grey_edge = int((1-p_augment) * 0.5 * 250)
        binary_map[:grey_edge, :] = 0
        binary_map[-grey_edge:, :] = 0
    elif p_augment < 0.333:
        grey_edge = int(p_augment * 0.5 * 250)
        binary_map[:, :grey_edge] = 0
        binary_map[:, -grey_edge:] = 0

    return np.clip(binary_map, 0, +1)
'''

def dict2array(origin_data):
    value_list = []
    if type(origin_data) is np.ndarray:
        return origin_data
    else:
        for key, value in origin_data.items():
            value_list.append(value)
        item = np.array(value_list[0]).flatten()
        for i in range(1, len(value_list)):
            item = np.append(item, np.array(value_list[i]).flatten())
        return item


def modify_action(action_dict):
    act_array = dict2array(action_dict)
    # print(act_array.shape)
    return act_array


def modify_obs(obs_dict):
    vehicle_pose = obs_dict['pose']                 # (x, y, z, r, p, y)
    vehicle_velocity = obs_dict['velocity']         # (vx, vy, vz, alpha_x, alpha_y, alpha_z)
    vehicle_accele = obs_dict['acceleration']      # (ax, ay, az, a_a_x, a_a_y, a_a_z)
    lidar_sensor = obs_dict['lidar']               # (1080 ray)

    roll_velocity, roll_accele = vehicle_velocity[-3], vehicle_accele[-3]
    pitch_velocity, pitch_accele = vehicle_velocity[-2], vehicle_accele[-2]
    yaw_velocity, yaw_accele = vehicle_velocity[-1], vehicle_accele[-1]
    
    abs_velocity = math.sqrt(vehicle_velocity[0]**2+vehicle_velocity[1]**2)
    abs_accele = math.sqrt(vehicle_accele[0]**2+vehicle_accele[1]**2)
    
    imu_sensor = np.array([roll_velocity, roll_accele, pitch_velocity, pitch_accele, yaw_velocity, yaw_accele, abs_velocity, abs_accele])
    output_obs = np.concatenate(([imu_sensor] + [lidar_sensor]))
    return output_obs

def Softmax_CL(demo_rew, agent_ep_rew):
    e_teacher, e_agent = math.exp((demo_rew+200)*0.005), math.exp(agent_ep_rew*0.005)
    teacher_power = (e_teacher)/(e_teacher+e_agent)
    agent_power = 1 - teacher_power
    return teacher_power, agent_power


def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

def keyboard_action(last_action):
    steering_flag, motor_flag = False, False
    left_sig, right_sig = keyboard.is_pressed('a'), keyboard.is_pressed('d')
    acc_sig, dea_sig = keyboard.is_pressed('w'), keyboard.is_pressed('s')

    if (left_sig and right_sig) or (not left_sig and not right_sig):
        steering_flag = False
        last_action[0] -= 0.25*last_action[0]
    else:
        steering_flag = True
        last_action[0] += (-0.05)*int(left_sig) + 0.05*int(right_sig)
    if (acc_sig and dea_sig) or (not acc_sig and not dea_sig):
        steering_flag = False
        last_action[1] -= 0.5*last_action[1]
    else:
        steering_flag = True
        last_action[1] += (-0.05)*int(dea_sig) + 0.05*int(acc_sig)

    last_action[0], last_action[1] =  last_action[0].round(3), last_action[1].round(3)
    output_action = np.clip(last_action, -1, +1)
    time.sleep(0.016)
    return output_action