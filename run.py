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


# Define training hyperparameters
SEED = 0
SOURCE_FOLDER = "input\\"

# Argument Parser
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--tester', default="log/run", required=False, type=str, help="Log folder.")
args = parser.parse_args()


def init_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    # Set up random seed on everything
    init_seed(SEED)

    
