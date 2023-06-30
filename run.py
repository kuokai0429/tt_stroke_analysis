# 2023.0613 @Brian

import os
import glob
import argparse
import pickle
import re
import random
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import fontManager
import cv2
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import math
from dtw import *
from tslearn.metrics import dtw_path, ctw_path, ctw
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler


def init_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def load_subject_keypoints(subject, annot_df):

    subject_annot = annot_df.loc[annot_df['subject'] == subject]

    subject_kp_filepath = f"common/pose3d/output/{subject}/keypoints_3d_mhformer.npz"
    assert os.path.exists(subject_kp_filepath), f"Subject {subject} 3D keypoints file doesn't exist!"

    subject_keypoints = np.load(subject_kp_filepath, encoding='latin1', allow_pickle=True)["reconstruction"]
    subject_trimmed_kp = subject_keypoints[int(subject_annot['start']):int(subject_annot['end'])]

    print(subject_trimmed_kp.shape)

    subject_video_filepath = f"input/{subject}.mp4"
    subject_video_fps = cv2.VideoCapture(subject_video_filepath).get(cv2.CAP_PROP_FPS)

    return subject_trimmed_kp, subject_video_fps


def distance(point_1, point_2):

    length = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2 + (point_1[2]-point_2[2])**2)
    return length


def calculateAngle(point_1, point_2, point_3):

    a = math.sqrt((point_2[0]-point_3[0])**2 + (point_2[1]-point_3[1])**2 + (point_2[2]-point_3[2])**2)
    b = math.sqrt((point_1[0]-point_3[0])**2 + (point_1[1]-point_3[1])**2 + (point_1[2]-point_3[2])**2)
    c = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2 + (point_1[2]-point_2[2])**2)

    if (-2*a*c) == 0:
        B = math.degrees(math.acos((b*b-a*a-c*c) / (-2*a*c+1)))
    elif ((b*b-a*a-c*c) / (-2*a*c)) >= 1 or ((b*b-a*a-c*c) / (-2*a*c)) <= -1:
        B = math.degrees(math.acos(1))
    else:
        B = math.degrees(round(math.acos((b*b-a*a-c*c) / (-2*a*c)), 3))

    return B


def get_curves(s1_time, s2_time, s1_feature, s2_feature):
    '''
    Get the interpolated curves of the original data.
    '''

    s1_xlim = np.arange(0, len(s1_time), 1)
    s1_x_curve = CubicSpline(s1_xlim, s1_time, bc_type='natural')
    s1_y_curve = CubicSpline(s1_xlim, s1_feature, bc_type='natural')

    s2_xlim = np.arange(0, len(s2_time), 1)
    s2_x_curve = CubicSpline(s2_xlim, s2_time, bc_type='natural')
    s2_y_curve = CubicSpline(s2_xlim, s2_feature, bc_type='natural')

    max_x, min_x = max(max(s1_time), max(s2_time)),  min(min(s1_time), min(s2_time))
    max_y, min_y = max(max(s1_feature), max(s2_feature)),  min(min(s1_feature), min(s2_feature))

    return s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y


def plot_subject_seperate(feature_name, xlabel, ylabel, s1_time, s2_time, s1_feature, s2_feature):

    s2_label = args.subject1 if args.mode == "benchmark" else args.subject2

    f = plt.figure()
    plt.xlim([min(min(s1_time), min(s2_time)), max(max(s1_time), max(s2_time))])
    plt.ylim([min(min(s1_feature), min(s2_feature)), max(max(s1_feature), max(s2_feature))])
    plt.plot(s1_time, s1_feature, label=args.subject1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#1f77b4")
    ax.get_lines()[0].set_color("#1f77b4")
    # plt.show()
    
    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_s1_{TIMESTAMP[:-1]}")

    f = plt.figure()
    plt.xlim([min(min(s1_time), min(s2_time)), max(max(s1_time), max(s2_time))])
    plt.ylim([min(min(s1_feature), min(s2_feature)), max(max(s1_feature), max(s2_feature))])
    plt.plot(s2_time, s2_feature, label=s2_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#ff7f0e")
    ax.get_lines()[0].set_color("#ff7f0e")
    # plt.show()

    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_s2_{TIMESTAMP[:-1]}")

    plt.close(f)


def plot_subject_concatenate(feature_name, xlabel, ylabel, s1_time, s2_time, s1_feature, s2_feature):

    s2_label = args.subject1 if args.mode == "benchmark" else args.subject2

    ## Plot Subjects together without 1d-rescale
    
    # f = plt.figure()
    # plt.plot(s1_time, s1_feature, label=args.subject1)
    # plt.plot(s2_time, s2_feature, label=s2_label)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/{TIMESTAMP}/{feature_name}_{TIMESTAMP[:-1]}")

    ## Plot Subjects together with 1d-rescale

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)

    factor = max_x / min(max(s1_time), max(s2_time))
    factor_s1, factor_s2 = (1, factor) if max(s1_time) > max(s2_time) else (factor, 1)

    f = plt.figure()
    plt.plot(factor_s1 * s1_x_curve(s1_xlim), s1_y_curve(s1_xlim), label=args.subject1)
    plt.plot(factor_s2 * s2_x_curve(s2_xlim), s2_y_curve(s2_xlim), label=s2_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.show()

    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_{TIMESTAMP[:-1]}")

    plt.close(f)


def plot_trajectory(ts, t, ax, color_code=None, alpha=.1):
    
    if color_code is not None:
        colors = [color_code] * len(ts)
    else:
        # colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
        colors = plt.cm.jet(5)
    for i in range(len(ts) - 1):
        ax.plot(ts[i:i+2], t[i:i+2],
                marker='o', c=colors, alpha=alpha)


def euclidean_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    https://tech.gorilla.co/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2, s_max = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)), np.random.uniform(min_y, max_y, size=1000).reshape(-1, 1)

    subject_distance = np.sqrt(np.sum((s1 - s2) ** 2))
    max_distance = np.sqrt(np.sum((s1 - s_max) ** 2))
    min_distance = 0

    similarity = (subject_distance / (max_distance - min_distance)) * 100
    similarity = min(max((100 - similarity), 0), 100)

    print(f"<Euclidean> Subject_distance: {subject_distance}, Max_distance: {max_distance}, Min_distance: {min_distance}")

    return similarity


def pearsonCorr_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2 = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000))

    a_diff = s1 - np.mean(s1)
    p_diff = s2 - np.mean(s2)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    subject_corr = numerator / denominator

    # print(f"Subject_Correlation: {subject_corr}")

    return min(max(subject_corr, 0) * 100, 100)


def dtw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    In time series analysis, dynamic time warping (DTW) is an algorithm for measuring similarity between 
    two temporal sequences, which may vary in speed. 
    (https://dynamictimewarping.github.io/python/)
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2, s_max = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)), np.random.uniform(min_y, max_y, size=1000).reshape(-1, 1)

    alignment_threeway = dtw(s1, s2, keep_internals=True)    
    alignment_twoway = dtw(s1, s2, keep_internals=True, open_begin=True, open_end=True, step_pattern=rabinerJuangStepPattern(6, "c"))
    alignment_threeway.plot(type="threeway")
    alignment_twoway.plot(type="twoway",offset=-2).figure.savefig(f"./output/{TIMESTAMP}/{feature_name}_similarity_{TIMESTAMP[:-1]}")
    # plt.show()
    
    subject_distance, min_distance, max_distance = alignment_twoway.distance, 0, dtw(s1, s_max, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).distance
    similarity = (subject_distance / (max_distance - min_distance)) * 100
    similarity = min(max((100 - similarity), 0), 100)

    print(f"<DTW> Subject_distance: {subject_distance}, Max_distance: {max_distance}, Min_distance: {min_distance}")

    return similarity


def meanStd_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):

    mean_standard, mean_learner = np.mean(s1_feature), np.mean(s2_feature)
    std_standard, std_learner  = np.std(s1_feature, ddof=0), np.std(s2_feature, ddof=1)

    if mean_learner < 160:
        mean = max((1 - (abs(mean_learner-mean_standard) / 180)) * 30, 0)
        std  = max((1 - (abs(std_learner-std_standard) / std_standard)) * 0, 0)
        bonus = 70
    else:
        mean = max((1 - (abs(mean_learner-mean_standard) / 180)) * 20, 0)
        std  = min(max((1 - (abs(std_learner-std_standard) / std_standard)) * 80, 0), 50)
        bonus = 0
        
    similarity = int(round(mean + std + bonus))

    return similarity


def ctw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    Canonical Time Warping is a method to align time series under rigid registration 
    of the feature space. It should not be confused with Dynamic Time Warping (DTW), though CTW uses DTW.
    https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.ctw.html
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2, s_max = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)), np.random.uniform(min_y, max_y, size=1000).reshape(-1, 1)

    path_dtw, _ = dtw_path(s1, s2)
    path_ctw, cca, _ = ctw_path(s1, s2, max_iter=100, n_components=1)
    # ctw_distance = ctw(s1, s2, max_iter=100, n_components=1)
    print(len(path_dtw), len(path_ctw))

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    for (i, j) in path_dtw:
        ax.plot([np.arange(0, 1000)[i], np.arange(0, 1000)[j]],
                [s1[i], s2[j]],
                color='g' if i == j else 'r', alpha=.5)
    plot_trajectory(np.arange(0, 1000), s1, ax)
    plot_trajectory(np.arange(0, 1000), s2, ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DTW")

    ax = plt.subplot(1, 2, 2)
    for (i, j) in path_ctw:
        ax.plot([np.arange(0, 1000)[i], np.arange(0, 1000)[j]],
                [s1[i], s2[j]],
                color='g' if i == j else 'r', alpha=.5)
    plot_trajectory(np.arange(0, 1000), s1, ax)
    plot_trajectory(np.arange(0, 1000), s2, ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("CTW")

    plt.tight_layout()
    plt.show()
    
    # subject_distance, min_distance, max_distance = , 0, 
    # similarity = (subject_distance / (max_distance - min_distance)) * 100
    # similarity = min(max((100 - similarity), 0), 100)

    # print(f"<CTW> Subject_distance: {subject_distance}, Max_distance: {max_distance}, Min_distance: {min_distance}")

    # return similarity


def similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((s1_feature, s2_feature), axis=0).reshape(-1, 1))
    s1_feature_scaled = scaler.transform(s1_feature.reshape(-1, 1))
    s2_feature_scaled = scaler.transform(s2_feature.reshape(-1, 1))
    s1_feature, s2_feature = s1_feature_scaled, s2_feature_scaled
    print(s1_feature_scaled.shape, s2_feature_scaled.shape)

    old_similarity = meanStd_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    euclidean_similarity = euclidean_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    pearsonCorr_similarity = pearsonCorr_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    dtw_similarity = dtw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    ctw_similarity = ctw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)

    if 99 <= pearsonCorr_similarity and 99 <= dtw_similarity:
        similarity = 100
    elif 99 <= pearsonCorr_similarity and dtw_similarity < 99:
        similarity = pearsonCorr_similarity - 0.2 * dtw_similarity
    elif pearsonCorr_similarity < 99 and 99 <= dtw_similarity:
        similarity = dtw_similarity - 0.2 * pearsonCorr_similarity
    elif 60 < pearsonCorr_similarity < 80 and 50 < dtw_similarity < 70:
        similarity = 0.6 * pearsonCorr_similarity + 0.4 * dtw_similarity + 15
    elif 60 < pearsonCorr_similarity < 80 and dtw_similarity < 50:
        similarity = 0.9 * pearsonCorr_similarity + 0.1 * dtw_similarity + 15
    elif 60 < pearsonCorr_similarity < 80 and 70 < dtw_similarity:
        similarity = 0.1 * pearsonCorr_similarity + 0.9 * dtw_similarity + 10
    elif pearsonCorr_similarity < 60 and 50 < dtw_similarity < 70:
        similarity = 0.4 * pearsonCorr_similarity + 0.6 * dtw_similarity + 25
    elif 80 < pearsonCorr_similarity and 50 < dtw_similarity < 70:
        similarity = 0.9 * pearsonCorr_similarity + 0.1 * dtw_similarity + 10
    else:
        similarity = max(pearsonCorr_similarity, dtw_similarity)

    similarity = min(similarity, 100)
    
    print('Euclidean similarity:', euclidean_similarity)
    print('Pearson Correlation similarity', pearsonCorr_similarity)
    print('DTW similarity:', dtw_similarity)
    print('CTW similarity:', ctw_similarity)
    print('Proposed similarity:', similarity, end="\n\n")

    with open(f'output/{TIMESTAMP}/{feature_name}_eval_{TIMESTAMP[:-1]}.txt', 'w') as f:
        
        f.writelines(f'<{feature_name}>\n')
        f.writelines(f'Euclidean similarity: {euclidean_similarity}\n')
        f.writelines(f'Pearson Correlation similarity: {pearsonCorr_similarity}\n')
        f.writelines(f'DTW similarity: {dtw_similarity}\n')
        f.writelines(f'CTW similarity: {ctw_similarity}\n')
        f.writelines(f'Proposed similarity: {similarity}\n')

    return similarity


def evaluate_arm_wave_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps, degree):

    print("Arm Waving Angles >>>>>")

    ## Configuring a Vector that points out from Front Body (Suppose the 3D Pose was facing right)

    vx, vy, vz = np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180), 0

    ## Calculate Subject Arm Waving Angles

    s1_strokes_arm_wave_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_strokes_kp])
    s2_strokes_arm_wave_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s2_strokes_kp])

    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]  

    ## Plot Subjects Arm Waving Angles
    
    plot_subject_seperate('arm_wave_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_arm_wave_ang, s2_strokes_arm_wave_ang)
    plot_subject_concatenate('arm_wave_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_arm_wave_ang, s2_strokes_arm_wave_ang)

    ## Calculate Subject Arm Waving Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('arm_wave_ang', s1_time, s2_time, s1_strokes_arm_wave_ang, s2_strokes_arm_wave_ang)

    return similarity


def evaluate_arm_bend_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps):

    print("Arm bending angle >>>>>")

    ## Calculate Subject Arm Bending Angles

    s1_strokes_arm_bend_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]], f[h36m_skeleton["r_wrist"]]) for f in s1_strokes_kp])
    s2_strokes_arm_bend_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]], f[h36m_skeleton["r_wrist"]]) for f in s2_strokes_kp])

    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]  

    ## Plot Subjects Arm Bending Angles
    
    plot_subject_seperate('arm_bend_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_arm_bend_ang, s2_strokes_arm_bend_ang)
    plot_subject_concatenate('arm_bend_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_arm_bend_ang, s2_strokes_arm_bend_ang)

    ## Calculate Subject Arm Bending Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('arm_bend_ang', s1_time, s2_time, s1_strokes_arm_bend_ang, s2_strokes_arm_bend_ang)

    return similarity


def evaluate_knee_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps):

    print("Knee bending angle >>>>>")

    ## Calculate Subject Knee Bending Angles

    s1_strokes_knee_ang = np.array([(calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["r_knee"]], f[h36m_skeleton["r_foot"]]) + 
                                    calculateAngle(f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_knee"]], f[h36m_skeleton["l_foot"]])) / 2 
                                    for f in s1_strokes_kp])
    s2_strokes_knee_ang = np.array([(calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["r_knee"]], f[h36m_skeleton["r_foot"]]) + 
                                    calculateAngle(f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_knee"]], f[h36m_skeleton["l_foot"]])) / 2
                                    for f in s2_strokes_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]  

    ## Plot Subjects Knee Bending Angles
    
    plot_subject_seperate('knee_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_knee_ang, s2_strokes_knee_ang)
    plot_subject_concatenate('knee_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_knee_ang, s2_strokes_knee_ang)

    ## Calculate Subject Knee Bending Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('knee_ang', s1_time, s2_time, s1_strokes_knee_ang, s2_strokes_knee_ang)

    return similarity


def evaluate_hip_rot_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps, degree):

    print("Hip Joint relative Angle >>>>>")

    ## Configuring a Vector that points out from Front Body (Suppose the 3D Pose was facing right)

    vx, vy, vz = np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180), 0

    ## Calculate Subject Hip Joint relative Angle

    s1_strokes_hip_rot_ang = np.array([calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_hip"]] + (vx, vy, vz)) for f in s1_strokes_kp])
    s2_strokes_hip_rot_ang = np.array([calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_hip"]] + (vx, vy, vz)) for f in s2_strokes_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]   

    ## Plot Subjects Hip Joint relative Angle
    
    plot_subject_seperate('hip_rot_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_hip_rot_ang, s2_strokes_hip_rot_ang)
    plot_subject_concatenate('hip_rot_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_hip_rot_ang, s2_strokes_hip_rot_ang)

    ## Calculate Subject Hip Joint relative Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('hip_rot_ang', s1_time, s2_time, s1_strokes_hip_rot_ang, s2_strokes_hip_rot_ang)

    return similarity


def evaluate_cog_trans(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps):

    print("Center of Gravity Transition >>>>>")

    ## Calculate Subject Center of Gravity Transitions

    s1_strokes_cog = np.array([(f[h36m_skeleton["spine"]] + f[h36m_skeleton["hip"]]) / 2 for f in s1_strokes_kp])
    s2_strokes_cog = np.array([(f[h36m_skeleton["spine"]] + f[h36m_skeleton["hip"]]) / 2 for f in s2_strokes_kp])

    s1_cog_trans = np.array([(distance(s1_strokes_cog[i], s1_strokes_cog[i-1]) / ((i/s1_video_fps) - ((i-1)/s1_video_fps)))
                     for i in range(1, len(s1_strokes_kp))])
    s2_cog_trans = np.array([(distance(s2_strokes_cog[i], s2_strokes_cog[i-1]) / ((i/s2_video_fps) - ((i-1)/s2_video_fps)))
                     for i in range(1, len(s2_strokes_kp))])
    
    s1_time = [(i / s1_video_fps) for i in range(1, len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(1, len(s2_strokes_kp))]  

    ## Plot Subjects Center of Gravity Transitions
    
    plot_subject_seperate('cog_trans', 'sec', 'velocity (pixel/s)', s1_time, s2_time, s1_cog_trans, s2_cog_trans)
    plot_subject_concatenate('cog_trans', 'sec', 'velocity (pixel/s)', s1_time, s2_time, s1_cog_trans, s2_cog_trans)

    ## Calculate Subject Center of Gravity Transitions Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('cog_trans', s1_time, s2_time, s1_cog_trans, s2_cog_trans)

    return similarity


def evaluate_strokes_speed(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps):

    print("Stroke Speed >>>>>")

    ## Calculate Subject Stroke Speed

    s1_strokes_wrist = np.array([f[h36m_skeleton["r_wrist"]] for f in s1_strokes_kp])
    s2_strokes_wrist = np.array([f[h36m_skeleton["r_wrist"]] for f in s2_strokes_kp])

    s1_strokes_speed = np.array([(distance(s1_strokes_wrist[i], s1_strokes_wrist[i-1]) / ((i/s1_video_fps) - ((i-1)/s1_video_fps)))
                     for i in range(1, len(s1_strokes_kp))])
    s2_strokes_speed = np.array([(distance(s2_strokes_wrist[i], s2_strokes_wrist[i-1]) / ((i/s2_video_fps) - ((i-1)/s2_video_fps)))
                     for i in range(1, len(s2_strokes_kp))])
    
    s1_time = [(i / s1_video_fps) for i in range(1, len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(1, len(s2_strokes_kp))]  

    ## Plot Subjects Stroke Speed
    
    plot_subject_seperate('stroke_speed', 'sec', 'velocity (pixel/s)', s1_time, s2_time, s1_strokes_speed, s2_strokes_speed)
    plot_subject_concatenate('stroke_speed', 'sec', 'velocity (pixel/s)', s1_time, s2_time, s1_strokes_speed, s2_strokes_speed)

    ## Calculate Subject Stroke Speed Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('stroke_speed', s1_time, s2_time, s1_strokes_speed, s2_strokes_speed)

    return similarity


def benchmark_comparison(s1_kp , s1_video_fps):

    print("Benchmark Comparison Between Similarity Functions:", end="\n")

    # Configurations
    degree, time_offset, value_offset, scale_value, noise_range, wrapped_time = 0, 10, 20, 1.5, 8, 20
    vx, vy, vz  = np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180), 0

    # Original
    print("Original----------------")
    s1_feature_original = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), 
                                                   f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_kp[:-time_offset]])
    s1_time_original = [(i / s1_video_fps) for i in range(len(s1_kp[:-time_offset]))]
    plot_subject_concatenate('benchmark_original', 'sec', 'degree', s1_time_original, s1_time_original, s1_feature_original, s1_feature_original)
    similarity = similarity_function('benchmark_original', s1_time_original, s1_time_original, s1_feature_original, s1_feature_original)
    
    # Time Offset
    print("Time Offset----------------")
    s1_feature_timeoffset = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), 
                                                     f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_kp[time_offset:]])
    s1_time_timeoffset = [(i / s1_video_fps) for i in range(len(s1_kp[time_offset:]))]
    plot_subject_concatenate('benchmark_timeoffset', 'sec', 'degree', s1_time_original, s1_time_timeoffset, s1_feature_original, s1_feature_timeoffset)
    similarity = similarity_function('benchmark_timeoffset', s1_time_original, s1_time_timeoffset, s1_feature_original, s1_feature_timeoffset)
    
    # Value Offset
    print("Value Offset----------------")
    s1_feature_valueoffset = np.array([value_offset + calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), 
                                                                     f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_kp[:-time_offset]])
    plot_subject_concatenate('benchmark_valueoffset', 'sec', 'degree', s1_time_original, s1_time_original, s1_feature_original, s1_feature_valueoffset)
    similarity = similarity_function('benchmark_valueoffset', s1_time_original, s1_time_original, s1_feature_original, s1_feature_valueoffset)
    
    # Scaled
    print("Scaled----------------")
    s1_feature_scaled = np.array([scale_value * calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), 
                                                               f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_kp[:-time_offset]])
    plot_subject_concatenate('benchmark_scaled', 'sec', 'degree', s1_time_original, s1_time_original, s1_feature_original, s1_feature_scaled)
    similarity = similarity_function('benchmark_scaled', s1_time_original, s1_time_original, s1_feature_original, s1_feature_scaled)
    
    # Noise
    print("Noise----------------")
    s1_feature_noise = np.random.normal(-noise_range,noise_range,len(s1_kp[:-time_offset])) + np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), 
                        f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_kp[:-time_offset]])
    plot_subject_concatenate('benchmark_noise', 'sec', 'degree', s1_time_original, s1_time_original, s1_feature_original, s1_feature_noise)
    similarity = similarity_function('benchmark_noise', s1_time_original, s1_time_original, s1_feature_original, s1_feature_noise)
    
    # Time Wrapped
    print("Time Wrapped----------------")
    s1_feature_wrapped = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]] + (vx, vy, vz), 
                                                  f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_kp[:-(time_offset+wrapped_time)]])
    s1_time_wrapped = [(i / s1_video_fps) for i in range(len(s1_kp[:-(time_offset+wrapped_time)]))]
    plot_subject_concatenate('benchmark_wrapped', 'sec', 'degree', s1_time_original, s1_time_wrapped, s1_feature_original, s1_feature_wrapped)
    similarity = similarity_function('benchmark_wrapped', s1_time_original, s1_time_wrapped, s1_feature_original, s1_feature_wrapped)
    


# Define Configurations
SEED = 0
SOURCE_FOLDER = "input\\"
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())

# Argument Parser
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--subject1', required=True, type=str, help="Subject 1 3D filename.")
parser.add_argument('--mode', required=True, type=str, help="Mode.")
args = parser.parse_known_args()[0]


if __name__ == "__main__":

    init_seed(SEED)

    h36m_skeleton = {
        "head": 10, "neck": 9, "throat": 8, "spine": 7, "hip": 0,
        "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16, "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
        "r_hip": 1, "r_knee": 2, "r_foot": 3, "l_hip": 4, "l_knee": 5, "l_foot": 6
        }

    subjects_annot_filepath = f"annotation/stroke_analysis.csv"
    assert os.path.exists(subjects_annot_filepath), "Subjects annotation file doesn't exist!"
    annot_df = pd.read_csv(subjects_annot_filepath, encoding='utf8')

    os.makedirs(f'output/{TIMESTAMP}')


    ## Stroke Analysis 

    if args.mode == "analysis":

        parser.add_argument('--subject2', required=True, type=str, help="Subject 2 3D filename.")
        args = parser.parse_args()

        # 0. Load subject strokes keypoints and Get subject video information
        
        s1_strokes_kp, s1_video_fps = load_subject_keypoints(args.subject1, annot_df)
        s2_strokes_kp, s2_video_fps = load_subject_keypoints(args.subject2, annot_df)
        print()

        # 1. Evaluate Arm waving angle
        arm_wave_ang_similarity = evaluate_arm_wave_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps, 0)

        # 2. Evaluate Arm bending angle
        arm_bend_ang_similarity = evaluate_arm_bend_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)

        # 3. Evaluate Knee bending angle
        knee_ang_similarity = evaluate_knee_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)

        # 4. Evaluate Hip joint rotating angle
        hip_rot_ang_similarity = evaluate_hip_rot_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps, 0)

        # 5. Evaluate Center of gravity transitions
        cog_trans_similarity = evaluate_cog_trans(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)

        # 6. Evaluate Speed of stroke
        strokes_speed_similarity = evaluate_strokes_speed(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)


        ## Export the Analysis Results

        with open(f'output/{TIMESTAMP}/evalAll_{TIMESTAMP[:-1]}.txt', 'w') as f:
            
            f.writelines(f'{args.subject1} & {args.subject2}\n')
            f.writelines(f'arm_wave_ang_similarity : {arm_wave_ang_similarity }\n')
            f.writelines(f'arm_bend_ang_similarity : {arm_bend_ang_similarity }\n')
            f.writelines(f'knee_ang_similarity: {knee_ang_similarity}\n')
            f.writelines(f'hip_rot_ang_similarity: {hip_rot_ang_similarity}\n')
            f.writelines(f'cog_trans_similarity: {cog_trans_similarity}\n')
            f.writelines(f'strokes_speed_similarity: {strokes_speed_similarity}\n')

    
    ## Benchmark Comparison Between Similarity Functions
    
    elif args.mode == "benchmark":

        s1_strokes_kp, s1_video_fps = load_subject_keypoints(args.subject1, annot_df)
        benchmark_comparison(s1_strokes_kp, s1_video_fps)