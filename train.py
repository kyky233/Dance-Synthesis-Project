# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import os
import math
import yaml
import argparse
import numpy as np
from data_utils.constants import GLOBAL_INFO_DIRECTORY, SAVE_YAML_FILE
from config import Config
from prediction_network import train_prediction


def get_npz_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.npy')]


def divide_time_factor(window):
    time_factor = []
    for win_index in range(window):
        t = win_index / (window - 1)
        time_factor.append(t)

    return np.array(time_factor)


def get_time_vector(window, dim=1):
    time_vector = []
    for win_index in range(window):
        t = win_index / (window - 1)
        time = np.ones(dim) * t
        time_vector.append(time)
    return np.array(time_vector)


def load_data(data_path, train_data_proportion):
    """
    data(npy): [frame_num, 226]  226 = 23 * 3 + 3 + 1 + 4 + 24 * 3 + 5 + 24 * 3
    include:
        joint_pos        # [joint_num, 3] - [23, 3]
        root_pos         # [3]
        root_rot         # [1]
        contact          # [4]
        velocity         # [joint_num, 3] - [23, 3]
        vel_factor       # [5]
        acceleration     # [joint_num, 3] - [24, 3]
    ps:
        joint_num: 24
    """
    all_files = get_npz_files(data_path)
    train_data_num = int(len(all_files) * train_data_proportion)
    npz_files = all_files[:train_data_num]
    data_set = []
    data_name = []
    file_num = len(npz_files)
    sum_frames = 0
    for i, bvh in enumerate(npz_files):
        strs = bvh.split("\\")
        data_name.append(strs[-1][:-4])
        print("load file %s (%d/%d)" % (strs[-1][:-4], i + 1, file_num))
        data = np.load(bvh)     # (600, 226)
        print("  shape:", data.shape)
        sum_frames += len(data)
        data_set.append(data[..., :-72])    # (600, 226-72)
    print("Load %d frames for train." % sum_frames)
    '''
    # data_set --- 23 * 3 + 3 + 1 + 4 + 24 * 3 + 5
        include:
        joint_pos        # [joint_num, 3] - [23, 3]
        root_pos         # [3]
        root_rot         # [1]
        contact          # [4]
        velocity         # [joint_num, 3] - [23, 3]
        vel_factor       # [5]
    '''
    return data_set, data_name


def get_parent_and_bone(filename):
    f = open(filename, "r")
    d = yaml.load(f, Loader=yaml.Loader)
    return d["parents"], d["bone_length"]


def get_random_noise(win, dim, noise_factor):
    rand = np.random.randn(win, dim) * noise_factor
    for i in range(1, len(rand) + 1):
        if win - i < 5:
            rand[i - 1, :] = 0
        elif 5 <= win - i < 30:
            factor = (win - i - 5) / 25
            rand[i - 1, :] = rand[i, :] * factor
    return rand


def train_prediction_network(args):
    """
    data_set(list of np, each np indicates a seq)
    data_set[i]  --- (n_frames, 154) --- 154 = 23 * 3 + 3 + 1 + 4 + 24 * 3 + 5
    include:
        joint_pos        # [joint_num, 3] - [23, 3]
        root_pos         # [3]
        root_rot         # [1]
        contact          # [4]
        velocity         # [joint_num, 3] - [23, 3]
        vel_factor       # [5]
    """
    config = Config()
    parents, bone_length = get_parent_and_bone(GLOBAL_INFO_DIRECTORY + SAVE_YAML_FILE)
    data_info = {}
    data_set, data_name = load_data(args.data_path, config.train_data_proportion)
    for win in range(config.p_min, config.p_max + 1):   # range(7, 73)
        win_step = math.ceil(config.win_step_factor * win)  # ceil ---  rounds a number UP to the nearest integer
        noise = get_random_noise(win - 1, config.lstm1_input_size, config.noise_factor) # (6, 1024)
        data_info[win] = [win_step, noise]
    data = np.load(GLOBAL_INFO_DIRECTORY + "mean_std.npz")
    mean, std = data["mean"], data["std"]  # 226, 226
    train_prediction(data_set, data_info, parents, bone_length, mean[:-72], std[:-72])


def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train", type=str, default='prediction')
    parser.add_argument("--data_path", type=str, default='data/Cyprus_out/')
    parser.add_argument("--predict_model_path", type=str, default=None)
    return parser.parse_args()


# --train prediction --data_path data/Cyprus_out/ --predict_model_path /
# --train prediction --data_path data/Cyprus_out/
if __name__ == '__main__':
    args = parse_args()
    if args.train == "prediction":
        train_prediction_network(args)
