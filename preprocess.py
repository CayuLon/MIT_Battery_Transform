# preprocess  minidata
# from paper 'Decentralized Deep Learning Approach for Lithium-Ion Batteries State of Health Forecasting Using Federated Learning'

import math
import os
import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


temperature_feature_interval = 1
current_feature_interval = 1

cap_outlier_diff_threshold = 0.015  # outlier detection

curve_ratio_min = 0.5
curve_ratio_max = 0.99
curve_ratio_steps = 9
curve_ratio_digits = 2
curve_ratio_step_size = (curve_ratio_max - curve_ratio_min) / curve_ratio_steps

data_dir_path = ''
batch1 = pickle.load(open(os.path.join(data_dir_path, 'batch1.pkl'), 'rb'))
batch2 = pickle.load(open(os.path.join(data_dir_path, 'batch2.pkl'), 'rb'))
batch3 = pickle.load(open(os.path.join(data_dir_path, 'batch3.pkl'), 'rb'))


batch = {**batch1, **batch2, **batch3}  # 合并三个字典


'''
 prepare for Feature 4 and Feature 5
'''
max_temps = []
min_temps = []
for key in batch.keys():
    max_temps.append(np.max(batch[key]['summary']['Tmax']))
    min_temps.append(np.min(batch[key]['summary']['Tmin'][np.nonzero(batch[key]['summary']['Tmin'])]))
    # np.nonzero() 函数返回非零元素的索引。
max_temp = np.max(max_temps)  # 所有电池的最大温度的最大温度
min_temp = np.min(min_temps)  # 所有电池的最小温度的最小温度

max_currents = []
min_currents = []
for key in batch.keys():
    for cycle in batch[key]['cycles'].keys():
        max_currents.append(np.max(batch[key]['cycles'][cycle]['I']))
        min_currents.append(np.min(batch[key]['cycles'][cycle]['I']))
max_current = np.max(max_currents)
min_current = np.min(min_currents)


pre_processed_data = {}


for key in batch.keys():

    ''' 
     Get Targets 
    '''
    caps = batch[key]['summary']['QC'][1:]

    # remove outliers
    cap_diff = np.abs(np.concatenate(([0], np.diff(caps))))
    valid_idx = np.where(cap_diff < cap_outlier_diff_threshold)  # 相邻点差值小的保留，去除了3个cycle
    caps = caps[valid_idx]

    # TODO: use different normalization??
    caps = caps / 1.1  # Qc == Capacity (SOH)

    ''' 
     Get Seven Features 
    '''
    cycles = batch[key]['cycles']
    feats = []
    for cycle_id in valid_idx[0]:  # valid_idx[0] 取出tuple里的唯一的ndarray索引，是cycles的子集
        cycle = cycles[str(cycle_id + 1)]
        time = cycle['t']
        current = cycle['I']
        temp = cycle['T']
        feat = [
            np.trapz((current == 0).astype(np.float32), time),  # Feature 3  # 计算的满足条件的有多少时间
            np.trapz((current > 0).astype(np.float32), time),  # Feature 1
            np.trapz((current < 0).astype(np.float32), time),  # Feature 2
            max(current),  # Feature 6
            abs(min(current))  # Feature 7
        ]
        ''' feature4:  Temperature intervals '''
        for t in range(math.floor(min_temp), math.ceil(max_temp), temperature_feature_interval):
            #  math.floor(min_temp), math.ceil(max_temp)；向下取整和向上取整
            feat = np.append(feat,
                             np.trapz(((temp > t) & (temp <= t + temperature_feature_interval)).astype(np.float32),
                                      time))  # np.trapz(...): 使用梯形法则计算积分；np.append 将结果添加到feat数组中。
        ''' feature5:  Charge current intervals '''
        # Charging Current is greater than zero
        for c in range(0, math.ceil(max_current), current_feature_interval):
            feat = np.append(feat,
                             np.trapz(((current > c) & (current <= c + current_feature_interval)).astype(np.float32),
                                      time))
        feats.append(feat)

    '''
     Random Curve Segment Ratios, < Each cell is different >
    '''
    curve_ratios = []
    random_min = curve_ratio_min  # curve_ratio_min = 0.5, Initial
    random_max = random_min + curve_ratio_step_size  # curve_ratio_step_size = 0.0544444444 (5%的范围内随机)
    for _ in range(curve_ratio_steps):  # curve_ratio_steps = 9
        new_ratio = round(random.uniform(random_min, min(random_max, curve_ratio_max)), curve_ratio_digits)  # curve_ratio_max = 0.99
        # round(x, n) 将 x 四舍五入到 n 位数
        curve_ratios.append(new_ratio)
        random_min = new_ratio
        random_max += curve_ratio_step_size
    curve_ratios.append(1.0)

    print('Preprocessed', key, 'with', len(feats), 'cycles')

    pre_processed_data[key] = {
        'capacities': torch.tensor(caps, dtype=torch.float32),
        'features': torch.tensor(feats, dtype=torch.float32),
        'curve_ratios': curve_ratios,
    }

pickle.dump(pre_processed_data, open(os.path.join(data_dir_path,'preprocessed_data.pkl'), 'wb'))
