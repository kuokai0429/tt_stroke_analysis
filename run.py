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
from scipy.interpolate import CubicSpline


def init_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def load_subject_strokes_keypoints(subject, annot_df):

    subject_annot = annot_df.loc[annot_df['subject'] == subject]

    subject_kp_filepath = f"common/pose3d/output/{subject}/keypoints_3d_mhformer.npz"
    assert os.path.exists(subject_kp_filepath), f"Subject {subject} 3D keypoints file doesn't exist!"

    subject_keypoints = np.load(subject_kp_filepath, encoding='latin1', allow_pickle=True)["reconstruction"]
    subject_strokes_kp = subject_keypoints[int(subject_annot['start']):int(subject_annot['end'])]

    print(subject_strokes_kp.shape)

    subject_video_filepath = f"input/{subject}.mp4"
    subject_video_fps = cv2.VideoCapture(subject_video_filepath).get(cv2.CAP_PROP_FPS)

    return subject_strokes_kp, subject_video_fps


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

    max_x, min_x = max(max(s1_time), max(s2_time)),  min(max(s1_time), max(s2_time))
    max_y, min_y = max(max(s1_feature), max(s2_feature)),  min(max(s1_feature), max(s2_feature))

    return s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y


def plot_subject_seperate(feature_name, xlabel, ylabel, s1_time, s2_time, s1_feature, s2_feature):

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
    plt.plot(s2_time, s2_feature, label=args.subject2)
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

    ## Plot Subjects together without 1d-rescale
    
    # f = plt.figure()
    # plt.plot(s1_time, s1_feature, label=args.subject1)
    # plt.plot(s2_time, s2_feature, label=args.subject2)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/{TIMESTAMP}/{feature_name}_{TIMESTAMP[:-1]}")

    ## Plot Subjects together with 1d-rescale

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)

    factor = max_x / min_x
    factor_s1, factor_s2 = (1, factor) if max(s1_time) > max(s2_time) else (factor, 1)

    f = plt.figure()
    plt.plot(factor_s1 * s1_x_curve(s1_xlim), s1_y_curve(s1_xlim), label=args.subject1)
    plt.plot(factor_s2 * s2_x_curve(s2_xlim), s2_y_curve(s2_xlim), label=args.subject2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.show()

    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_{TIMESTAMP[:-1]}")

    plt.close(f)


def euclidean_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    https://tech.gorilla.co/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2 = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000))
    
    subject_distance = np.sqrt(np.sum((s1 - s2) ** 2))
    max_distance = np.sqrt(np.sum((s1 - np.random.randint(min_y, max_y+1, size=1000)) ** 2))
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
    subject_corr = abs(numerator / denominator)

    # print(f"Subject_Correlation: {subject_corr}")

    return subject_corr


def dtw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    In time series analysis, dynamic time warping (DTW) is an algorithm for measuring similarity between 
    two temporal sequences, which may vary in speed. 
    (https://dynamictimewarping.github.io/python/)
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)

    alignment_threeway = dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)),
        keep_internals=True)    
    alignment_twoway = dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)),
        keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
      
    alignment_threeway.plot(type="threeway")
    alignment_twoway.plot(type="twoway",offset=-2).figure.savefig(f"./output/{TIMESTAMP}/{feature_name}_similarity_{TIMESTAMP[:-1]}")
    # plt.show()
    
    subject_distance, min_distance, max_distance = alignment_twoway.distance, 0, dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), np.random.randint(min_y, max_y+1, size=1000),
        keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).distance
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


def similarity_function(feature_name, s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang):

    old_similarity = meanStd_similarity_function(feature_name, s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)
    euclidean_similarity = euclidean_similarity_function(feature_name, s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)
    pearsonCorr_similarity = pearsonCorr_similarity_function(feature_name, s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)
    dtw_similarity = dtw_similarity_function(feature_name, s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)
    similarity = 0.5 * 100 * pearsonCorr_similarity + 0.5 * dtw_similarity
    
    print("Old similarity: ", old_similarity)
    print('Euclidean similarity:', euclidean_similarity)
    print('Pearson Correlation similarity', pearsonCorr_similarity)
    print('DTW similarity:', dtw_similarity)
    print('similarity:', similarity, end="\n\n")

    return similarity


def evaluate_arm_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps):

    print("Arm bending angle >>>>>")

    ## Calculate Subject Arm Bending Angles

    # s1_strokes_arm_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]], f[h36m_skeleton["r_wrist"]]) for f in s1_strokes_kp])
    # s2_strokes_arm_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]], f[h36m_skeleton["r_wrist"]]) for f in s2_strokes_kp])

    s1_strokes_arm_ang = np.array([calculateAngle(f[h36m_skeleton["throat"]], f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s1_strokes_kp])
    s2_strokes_arm_ang = np.array([calculateAngle(f[h36m_skeleton["throat"]], f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]]) for f in s2_strokes_kp])

    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]  

    ## Plot Subjects Arm Bending Angles
    
    plot_subject_seperate('arm_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)
    plot_subject_concatenate('arm_ang', 'sec', 'degree', s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)

    ## Calculate Subject Arm Bending Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('arm_ang', s1_time, s2_time, s1_strokes_arm_ang, s2_strokes_arm_ang)

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


def evaluate_hip_rot_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps, pose_frontfacing_degree):

    print("Hip Joint relative Angle >>>>>")

    ## Calculate Subject Hip Joint relative Angle

    vz, vx, vy = 0, np.cos(pose_frontfacing_degree * np.pi / 180), np.sin(pose_frontfacing_degree * np.pi / 180)
    s1_strokes_hip_rot_ang = np.array([calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_hip"]] + (vz, vx, vy)) for f in s1_strokes_kp])
    s2_strokes_hip_rot_ang = np.array([calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_hip"]] + (vz, vx, vy)) for f in s2_strokes_kp])
    
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

    s1_cog_trans = [(distance(s1_strokes_cog[i], s1_strokes_cog[i-1]) / ((i/s1_video_fps) - ((i-1)/s1_video_fps)))
                     for i in range(1, len(s1_strokes_kp))]
    s2_cog_trans = [(distance(s2_strokes_cog[i], s2_strokes_cog[i-1]) / ((i/s2_video_fps) - ((i-1)/s2_video_fps)))
                     for i in range(1, len(s2_strokes_kp))]
    
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

    s1_strokes_speed = [(distance(s1_strokes_wrist[i], s1_strokes_wrist[i-1]) / ((i/s1_video_fps) - ((i-1)/s1_video_fps)))
                     for i in range(1, len(s1_strokes_kp))]
    s2_strokes_speed = [(distance(s2_strokes_wrist[i], s2_strokes_wrist[i-1]) / ((i/s2_video_fps) - ((i-1)/s2_video_fps)))
                     for i in range(1, len(s2_strokes_kp))]
    
    s1_time = [(i / s1_video_fps) for i in range(1, len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(1, len(s2_strokes_kp))]  

    ## Plot Subjects Stroke Speed
    
    plot_subject_seperate('stroke_speed', 'sec', 'velocity (pixel/s)', s1_time, s2_time, s1_strokes_speed, s2_strokes_speed)
    plot_subject_concatenate('stroke_speed', 'sec', 'velocity (pixel/s)', s1_time, s2_time, s1_strokes_speed, s2_strokes_speed)

    ## Calculate Subject Stroke Speed Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('stroke_speed', s1_time, s2_time, s1_strokes_speed, s2_strokes_speed)

    return similarity


# Define Configurations
SEED = 0
SOURCE_FOLDER = "input\\"
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())

# Argument Parser
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--subject1', required=True, type=str, help="Subject 1 3D filename.")
parser.add_argument('--subject2', required=True, type=str, help="Subject 2 3D filename.")
args = parser.parse_args()


if __name__ == "__main__":

    ## Set up random seed on everything

    init_seed(SEED)


    ## Load Annotation file

    subjects_annot_filepath = f"annotation/stroke_analysis.csv"
    assert os.path.exists(subjects_annot_filepath), "Subjects annotation file doesn't exist!"
    annot_df = pd.read_csv(subjects_annot_filepath, encoding='utf8')


    ## Load subject strokes keypoints and Get subject video information

    h36m_skeleton = {
        "head": 10, "neck": 9, "throat": 8, "spine": 7, "hip": 0,
        "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16, "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
        "r_hip": 1, "r_knee": 2, "r_foot": 3, "l_hip": 4, "l_knee": 5, "l_foot": 6
        }
    
    s1_strokes_kp, s1_video_fps = load_subject_strokes_keypoints(args.subject1, annot_df)
    s2_strokes_kp, s2_video_fps = load_subject_strokes_keypoints(args.subject2, annot_df)
    print()


    ## Stroke Analysis 

    os.makedirs(f'output/{TIMESTAMP}')

    # 1. Evaluate Arm bending angle
    arm_ang_similarity = evaluate_arm_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)

    # 2. Evaluate Knee bending angle
    knee_ang_similarity = evaluate_knee_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)

    # 3. Evaluate Hip joint rotation angle
    hip_rot_ang_similarity = evaluate_hip_rot_ang(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps, 180)

    # 4. Evaluate Center of gravity transitions
    cog_trans_similarity = evaluate_cog_trans(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)

    # 5. Evaluate Speed of stroke
    strokes_speed_similarity = evaluate_strokes_speed(s1_strokes_kp, s2_strokes_kp, s1_video_fps, s2_video_fps)