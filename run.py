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
from numpy import sin
from numpy import sqrt
from numpy import arange
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


def distance(point_1, point_2):

    length = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2 + (point_1[2]-point_2[2])**2)
    return length


def calculateAngle(point1, point2, point3):
                        
    # Find direction vector of line AB-> & BC->
    ABx, ABy, ABz = point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]
    BCx, BCy, BCz = point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]
 
    # Find the dotProduct of lines AB-> & BC->
    dotProduct = (ABx * BCx + ABy * BCy + ABz * BCz)
 
    # Find magnitude of line AB-> and BC->
    magnitudeAB = (ABx * ABx + ABy * ABy + ABz * ABz)
    magnitudeBC = (BCx * BCx + BCy * BCy + BCz * BCz)
 
    # Find the cosine of the angle formed by line AB-> and BC->
    angle = dotProduct / math.sqrt(magnitudeAB * magnitudeBC)
 
    # Find angle in radian
    angle = (angle * 180) / 3.14
 
    return round(abs(angle), 4)


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


def evaluate_arm_ang(s1_strokes_kp, s2_strokes_kp):

    ## Calculate Subject Arm Bending Angles

    s1_strokes_arm_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]], f[h36m_skeleton["r_wrist"]]) for f in s1_strokes_kp])
    s2_strokes_arm_ang = np.array([calculateAngle(f[h36m_skeleton["r_shoulder"]], f[h36m_skeleton["r_elbow"]], f[h36m_skeleton["r_wrist"]]) for f in s2_strokes_kp])

    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]  

    ## Plot Subjects seperately

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_strokes_arm_ang), max(s2_strokes_arm_ang))])
    plt.plot(s1_time, s1_strokes_arm_ang, label=args.subject1)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#1f77b4")
    ax.get_lines()[0].set_color("#1f77b4")
    plt.show()
    
    f.savefig(f"./output/arm_ang_s1_{TIMESTAMP[:-1]}")

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_strokes_arm_ang), max(s2_strokes_arm_ang))])
    plt.plot(s2_time, s2_strokes_arm_ang, label=args.subject2)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#ff7f0e")
    ax.get_lines()[0].set_color("#ff7f0e")
    plt.show()

    f.savefig(f"./output/arm_ang_s2_{TIMESTAMP[:-1]}")

    ## Plot Subjects together

    # f = plt.figure()
    # plt.plot(s1_time, s1_strokes_arm_ang, label=args.subject1)
    # plt.plot(s2_time, s2_strokes_arm_ang, label=args.subject2)
    # plt.xlabel('sec')
    # plt.ylabel('arm angle')
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/arm_ang_{TIMESTAMP[:-1]}")

    ## Calculate Subject Arm Bending Angles Similarities

    s1_mean, s1_std = np.mean(s1_strokes_arm_ang), np.std(s1_strokes_arm_ang, ddof=0)
    s2_mean, s2_std = np.mean(s2_strokes_arm_ang), np.std(s2_strokes_arm_ang, ddof=1)

    mean_error = max((1 - (abs(s1_mean - s2_mean)/180)) * 20, 0) # 0 <= mean_error
    std_error  = max(min((1 - (abs(s1_std - s2_std)/s2_std)) * 80, 50), 0) # 0 <= std_error <= 50

    similarity = int(round(mean_error + std_error))

    print('Arm bending angle similarity:', similarity)

    return similarity


def evaluate_knee_ang(s1_strokes_kp, s2_strokes_kp):

    ## Calculate Subject Knee Bending Angles

    s1_strokes_knee_ang = np.array([(calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["r_knee"]], f[h36m_skeleton["r_foot"]]) + 
                                    calculateAngle(f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_knee"]], f[h36m_skeleton["l_foot"]])) / 2
                                    for f in s1_strokes_kp])
    s2_strokes_knee_ang = np.array([(calculateAngle(f[h36m_skeleton["r_hip"]], f[h36m_skeleton["r_knee"]], f[h36m_skeleton["r_foot"]]) + 
                                    calculateAngle(f[h36m_skeleton["l_hip"]], f[h36m_skeleton["l_knee"]], f[h36m_skeleton["l_foot"]])) / 2
                                    for f in s2_strokes_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_strokes_kp))]  

    ## Plot Subjects seperately

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_strokes_knee_ang), max(s2_strokes_knee_ang))])
    plt.plot(s1_time, s1_strokes_knee_ang, label=args.subject1)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#1f77b4")
    ax.get_lines()[0].set_color("#1f77b4")
    plt.show()
    
    f.savefig(f"./output/knee_ang_s1_{TIMESTAMP[:-1]}")

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_strokes_knee_ang), max(s2_strokes_knee_ang))])
    plt.plot(s2_time, s2_strokes_knee_ang, label=args.subject2)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#ff7f0e")
    ax.get_lines()[0].set_color("#ff7f0e")
    plt.show()

    f.savefig(f"./output/knee_ang_s2_{TIMESTAMP[:-1]}")

    ## Plot Subjects together

    # f = plt.figure()
    # plt.plot(s1_time, s1_strokes_knee_ang, label=args.subject1)
    # plt.plot(s2_time, s2_strokes_knee_ang, label=args.subject2)
    # plt.xlabel('sec')
    # plt.ylabel('knee angle')
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/knee_ang_{TIMESTAMP[:-1]}")

    ## Calculate Subject Knee Bending Angles Similarities

    s1_mean, s1_std = np.mean(s1_strokes_knee_ang), np.std(s1_strokes_knee_ang, ddof=0)
    s2_mean, s2_std = np.mean(s2_strokes_knee_ang), np.std(s2_strokes_knee_ang, ddof=1)

    mean_error = max((1 - (abs(s1_mean - s2_mean)/180)) * 20, 0) # 0 <= mean_error
    std_error  = max(min((1 - (abs(s1_std - s2_std)/s2_std)) * 80, 50), 0) # 0 <= std_error <= 50

    similarity = int(round(mean_error + std_error))

    print('Knee bending angle similarity:', similarity)

    return similarity


def evaluate_cog_trans(s1_strokes_kp, s2_strokes_kp):

    ## Calculate Subject Center of Gravity Transitions

    s1_strokes_cog = np.array([(f[h36m_skeleton["spine"]] + f[h36m_skeleton["hip"]]) / 2 for f in s1_strokes_kp])
    s2_strokes_cog = np.array([(f[h36m_skeleton["spine"]] + f[h36m_skeleton["hip"]]) / 2 for f in s2_strokes_kp])

    s1_cog_trans = [(distance(s1_strokes_cog[i], s1_strokes_cog[i-1]) / ((i/s1_video_fps) - ((i-1)/s1_video_fps)))
                     for i in range(1, len(s1_strokes_kp))]
    s2_cog_trans = [(distance(s2_strokes_cog[i], s2_strokes_cog[i-1]) / ((i/s2_video_fps) - ((i-1)/s2_video_fps)))
                     for i in range(1, len(s2_strokes_kp))]
    
    s1_time = [(i / s1_video_fps) for i in range(1, len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(1, len(s2_strokes_kp))]  

    ## Plot Subjects seperately

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_cog_trans), max(s2_cog_trans))])
    plt.plot(s1_time, s1_cog_trans, label=args.subject1)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#1f77b4")
    ax.get_lines()[0].set_color("#1f77b4")
    plt.show()
    
    f.savefig(f"./output/cog_trans_s1_{TIMESTAMP[:-1]}")

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_cog_trans), max(s2_cog_trans))])
    plt.plot(s2_time, s2_cog_trans, label=args.subject2)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#ff7f0e")
    ax.get_lines()[0].set_color("#ff7f0e")
    plt.show()

    f.savefig(f"./output/cog_trans_s2_{TIMESTAMP[:-1]}")

    ## Plot Subjects together

    # f = plt.figure()
    # plt.plot(s1_time, s1_cog_trans, label=args.subject1)
    # plt.plot(s2_time, s2_cog_trans, label=args.subject2)
    # plt.xlabel('sec')
    # plt.ylabel('velocity (pixel/s)')
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/cog_trans_{TIMESTAMP[:-1]}")

    ## Calculate Subject Center of Gravity Transition Similarities

    s1_mean, s1_std = np.mean(s1_cog_trans), np.std(s1_cog_trans, ddof=0)
    s2_mean, s2_std = np.mean(s2_cog_trans), np.std(s2_cog_trans, ddof=1)

    mean_error = max((1 - (abs(s1_mean - s2_mean)/s2_mean*1.3)) * 20, 0) # 0 <= mean_error
    std_error  = max((1 - (abs(s1_std - s2_std)/s2_std*1.3)) * 80, 0) # 0 <= std_error

    similarity = int(round(mean_error + std_error))

    print('Center of Gravity Transition similarity:', similarity)

    return similarity


def evaluate_strokes_speed(s1_strokes_kp, s2_strokes_kp):

    ## Calculate Subject Stroke Speed

    s1_strokes_wrist = np.array([f[h36m_skeleton["r_wrist"]] for f in s1_strokes_kp])
    s2_strokes_wrist = np.array([f[h36m_skeleton["r_wrist"]] for f in s2_strokes_kp])

    s1_strokes_speed = [(distance(s1_strokes_wrist[i], s1_strokes_wrist[i-1]) / ((i/s1_video_fps) - ((i-1)/s1_video_fps)))
                     for i in range(1, len(s1_strokes_kp))]
    s2_strokes_speed = [(distance(s2_strokes_wrist[i], s2_strokes_wrist[i-1]) / ((i/s2_video_fps) - ((i-1)/s2_video_fps)))
                     for i in range(1, len(s2_strokes_kp))]
    
    s1_time = [(i / s1_video_fps) for i in range(1, len(s1_strokes_kp))]
    s2_time = [(i / s2_video_fps) for i in range(1, len(s2_strokes_kp))]  

    ## Plot Subjects seperately

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_strokes_speed), max(s2_strokes_speed))])
    plt.plot(s1_time, s1_strokes_speed, label=args.subject1)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#1f77b4")
    ax.get_lines()[0].set_color("#1f77b4")
    # plt.show()
    
    # f.savefig(f"./output/stroke_speed_s1_{TIMESTAMP[:-1]}")

    f = plt.figure()
    plt.xlim([0, max(max(s1_time), max(s2_time))])
    plt.ylim([0, max(max(s1_strokes_speed), max(s2_strokes_speed))])
    plt.plot(s2_time, s2_strokes_speed, label=args.subject2)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#ff7f0e")
    ax.get_lines()[0].set_color("#ff7f0e")
    # plt.show()

    # f.savefig(f"./output/stroke_speed_s2_{TIMESTAMP[:-1]}")

    ## Plot Subjects together without 1d-rescale
    
    # f = plt.figure()
    # plt.plot(s1_time, s1_strokes_speed, label=args.subject1)
    # plt.plot(s2_time, s2_strokes_speed, label=args.subject2)
    # plt.xlabel('sec')
    # plt.ylabel('velocity (pixel/s)')
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/stroke_speed_{TIMESTAMP[:-1]}")

    ## Plot Subjects together with 1d-rescale

    s1_x_line = arange(0, len(s1_time), 1)
    s1_x_curve = CubicSpline(s1_x_line, s1_time, bc_type='natural')
    s1_y_curve = CubicSpline(s1_x_line, s1_strokes_speed, bc_type='natural')

    s2_x_line = arange(0, len(s2_time), 1)
    s2_x_curve = CubicSpline(arange(0, len(s2_time), 1), s2_time, bc_type='natural')
    s2_y_curve = CubicSpline(arange(0, len(s2_time), 1), s2_strokes_speed, bc_type='natural')

    max_x, min_x = max(max(s1_time), max(s2_time)),  min(max(s1_time), max(s2_time))
    max_y, min_y = max(max(s1_strokes_speed), max(s2_strokes_speed)),  min(max(s1_strokes_speed), max(s2_strokes_speed))
    factor = max_x / min_x
    factor_s1, factor_s2 = (1, factor) if max(s1_time) > max(s2_time) else (factor, 1)

    f = plt.figure()
    plt.plot(factor_s1 * s1_x_curve(s1_x_line), s1_y_curve(s1_x_line), label=args.subject1)
    plt.plot(factor_s2 * s2_x_curve(s2_x_line), s2_y_curve(s2_x_line), label=args.subject2)
    plt.xlabel('sec')
    plt.ylabel('velocity (pixel/s)')
    plt.legend()
    # plt.show()

    # f.savefig(f"./output/stroke_speed_{TIMESTAMP[:-1]}")

    ## Calculate Subject Stroke Speed Similarities ver1

    s1_mean, s1_std = np.mean(s1_strokes_speed), np.std(s1_strokes_speed, ddof=0)
    s2_mean, s2_std = np.mean(s2_strokes_speed), np.std(s2_strokes_speed, ddof=1)

    mean_error = max((1 - (abs(s1_mean - s2_mean)/s2_mean)) * 20, 0) # 0 <= mean_error
    std_error  = max((1 - (abs(s1_std - s2_std)/s2_std)) * 80, 0) # 0 <= std_error

    similarity_1 = int(round(mean_error + std_error))

    print('Stroke Speed similarity ver1: ', similarity_1)

    ## Calculate Subject Stroke Speed Similarities ver2 (https://dynamictimewarping.github.io/python/)

    dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)),
        keep_internals=True).plot(type="threeway")
    # alignment = dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)),
    #     keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
    alignment = dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), s1_y_curve(np.linspace(0, len(s1_time), 1000)),
        keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
    alignment.plot(type="twoway",offset=-2)
    plt.show()
    
    subject_distance = alignment.distance
    max_distance = dtw(s1_y_curve(np.linspace(0, len(s1_time), 1000)), np.random.randint(0, max_y+1, size=1000),
        keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).distance

    # Apply logarithmic transformation to the DTW distance
    transformed_distance = math.log(subject_distance + 1)
    max_transformed = math.log(max_distance + 1)
    min_transformed = math.log(1)
    print(subject_distance, transformed_distance, max_transformed, min_transformed)

    # Normalize the transformed distance to a similarity score percentage
    similarity_2 = (transformed_distance / (max_transformed - min_transformed)) * 100
    similarity_2 = 100 - similarity_2 

    print('Stroke Speed similarity ver2: ', similarity_2)

    return similarity_1, similarity_2


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


    ## Stroke Analysis 

    # 1. Evaluate Arm bending angle
    # arm_ang_similarity = evaluate_arm_ang(s1_strokes_kp, s2_strokes_kp)

    # 2. Evaluate Knee bending angle
    # knee_ang_similarity = evaluate_knee_ang(s1_strokes_kp, s2_strokes_kp)

    # 3. Evaluate Hip joint rotation angle
    # hip_rot_ang_similarity =

    # 4. Evaluate Center of gravity transitions
    # cog_trans_similarity = evaluate_cog_trans(s1_strokes_kp, s2_strokes_kp)

    # 5. Evaluate Speed of stroke
    strokes_speed_similarity = evaluate_strokes_speed(s1_strokes_kp, s2_strokes_kp)

