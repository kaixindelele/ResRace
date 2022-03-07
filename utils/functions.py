import math
import random
import numpy as np
import keyboard
import sys
import tty
import termios
import time

imu_scale = 1
lidar_scale = 1

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


def modify_obs(obs_dict, a=np.array([0, 0]), lidar_switch=1):
    # vehicle_pose = obs_dict['pose']                 # (x, y, z, r, p, y)
    vehicle_velocity = imu_scale*obs_dict['velocity']         # (vx, vy, vz, alpha_x, alpha_y, alpha_z)
    vehicle_accele = imu_scale*obs_dict['acceleration']      # (ax, ay, az, a_a_x, a_a_y, a_a_z)
    lidar_sensor = lidar_scale*obs_dict['lidar']               # (1080 ray)
    roll_velocity, roll_accele = vehicle_velocity[-3], vehicle_accele[-3]
    pitch_velocity, pitch_accele = vehicle_velocity[-2], vehicle_accele[-2]
    yaw_velocity, yaw_accele = vehicle_velocity[-1], vehicle_accele[-1]
    abs_velocity = math.sqrt(vehicle_velocity[0]**2+vehicle_velocity[1]**2)
    abs_accele = math.sqrt(vehicle_accele[0]**2+vehicle_accele[1]**2)
    # imu_sensor = [vehicle_velocity[0], vehicle_velocity[1], vehicle_velocity[-1], vehicle_accele[0], vehicle_accele[1]]
    imu_sensor = np.array([roll_velocity, roll_accele, pitch_velocity, pitch_accele, yaw_velocity, yaw_accele, abs_velocity, abs_accele])
    # imu_velo = [vehicle_velocity[0], vehicle_velocity[1], vehicle_velocity[-1]]
    if lidar_switch == 1:
        output_obs = np.concatenate(([a] + [imu_sensor] + [lidar_sensor]))
    else:
        output_obs = np.concatenate(([a] + [imu_sensor]))
    # if train_mode == 'SAC':
        # output_obs = np.concatenate(([action_array]+[imu_sensor]+[lidar_sensor]))
    # elif train_mode == 'DRQ':
        # output_obs = np.concatenate(([action_array]+[imu_sensor]+[lidar2binary(lidar_sensor).ravel()]))
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
    left_sig, right_sig = keyboard.is_pressed('left'), keyboard.is_pressed('right')
    acc_sig, dea_sig = keyboard.is_pressed('up'), keyboard.is_pressed('down')

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
    return output_action


# def addtional_reward(origin_rew, current_lap, last_lap, lap_rew, track, delta_thres=0.005):




def progress_filter(last_progress, current_progress, lap_counter, delta_thres=0.005):
    finish_lap = False
    if last_progress >= 0.95 and current_progress < 0.05 and lap_counter >= 450:
        finish_lap = True
    delta_progress = 0 if abs(current_progress-last_progress) >= delta_thres else(current_progress-last_progress)
    return finish_lap, delta_progress
        
    
