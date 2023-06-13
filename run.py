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


def init_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


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


# Define Configurations
SEED = 0
SOURCE_FOLDER = "input\\"

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


    ## Load Subject keypoints

    s1_kp_filepath = f"common/pose3d/output/{args.subject1}/keypoints_3d_mhformer.npz"
    s2_kp_filepath = f"common/pose3d/output/{args.subject2}/keypoints_3d_mhformer.npz"

    assert os.path.exists(s1_kp_filepath), "Subject1 3D keypoints file doesn't exist!"
    assert os.path.exists(s2_kp_filepath), "Subject2 3D keypoints file doesn't exist!"

    s1_keypoints = np.load(s1_kp_filepath, encoding='latin1', allow_pickle=True)["reconstruction"]
    s2_keypoints = np.load(s2_kp_filepath, encoding='latin1', allow_pickle=True)["reconstruction"]
    

    ## Stroke Analysis Configurations
    
    human36m_skeleton = {
        "head": 10, "neck": 9, "throat": 8, "spine": 7, "hip": 0,
        "r_shoulder": 14, "r_albow": 15, "r_wrist": 16, "l_shoulder": 11, "l_albow": 12, "l_wrist": 13,
        "r_hip": 1, "r_knee": 2, "r_foot": 3, "l_hip": 4, "l_knee": 5, "l_foot": 6
        }
    
    s1_annot = annot_df.loc[annot_df['subject'] == args.subject1]
    s1_strokes_kp = s1_keypoints[int(s1_annot['start']):int(s1_annot['end'])]

    s2_annot = annot_df.loc[annot_df['subject'] == args.subject2]
    s2_strokes_kp = s2_keypoints[int(s2_annot['start']):int(s2_annot['end'])]

    print(len(s1_strokes_kp), len(s2_strokes_kp))


    ## 1. Evaluate Arm bending angle

    

    ## 2. Evaluate Knee bending angle



    ## 3. Evaluate Hip joint rotation angle



    ## 4. Evaluate Center of gravity transitions
