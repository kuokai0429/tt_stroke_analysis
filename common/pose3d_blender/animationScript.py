# 2023.0618 @Brian
# Simple animation scripts.

import numpy as np
import bpy

import sys
import subprocess
import os
import csv


def load_subject_strokes_keypoints(subject, subjects_annot_filepath):

#    subject_annot = annot_df.loc[annot_df['subject'] == subject]

    column_name = {'subject': 0, 'start': 1, 'end': 2, 'num_strokes': 3, 'stroke_class': 4}
    
    with open(subjects_annot_filepath, 'r') as file:
        csvreader = csv.reader(file) 
        for row in csvreader:
            if row[0] == subject:
                subject_annot = row
                
    print(subject_annot)

    subject_kp_filepath = f"common/pose3d/output/{subject}/keypoints_3d_mhformer.npz"
    assert os.path.exists(subject_kp_filepath), f"Subject {subject} 3D keypoints file doesn't exist!"

    subject_keypoints = np.load(subject_kp_filepath, encoding='latin1', allow_pickle=True)["reconstruction"]
    subject_strokes_kp = subject_keypoints[int(subject_annot[column_name['start']]):int(subject_annot[column_name['end']])]

    print(subject_strokes_kp.shape)

    return subject_strokes_kp


def changeCubeLocation(cube, x, y, z, frame):
    
    # Change the location of the cube
    cube.location.x = x
    cube.location.y = y
    cube.location.z = z

    # Insert keyframe at the last frame
    end_frame = frame
    cube.keyframe_insert("location", frame=end_frame)
    
    
if __name__ == "__main__":
    
    
    ## Change OS working directory
    
    os.chdir(r"C:\Users\user\Desktop\stroke_analysis")
    print(os.getcwd())
    
    
    ## Verify Annotation file

    subjects_annot_filepath = f"annotation/stroke_analysis.csv"
    assert os.path.exists(subjects_annot_filepath), "Subjects annotation file doesn't exist!"


    ## Load subject strokes keypoints and Get subject video information
                 
    h36m_skeleton = {
        "head": 10, "neck": 9, "throat": 8, "spine": 7, "hip": 0,
        "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16, "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
        "r_hip": 1, "r_knee": 2, "r_foot": 3, "l_hip": 4, "l_knee": 5, "l_foot": 6
        }
    
    s1_strokes_kp = load_subject_strokes_keypoints("other_f1_left", subjects_annot_filepath)
    
    
    ## Remove Objects from scene
    
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    
    ## Add Light, Camera, Plane and 17 keypoints cube into the scene
    
    light = bpy.data.objects.new('Light', bpy.data.lights.new('light', type='POINT'))
    bpy.context.collection.objects.link(light)
    light.location = (25, 5, 20)
    
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('camera'))
    bpy.context.collection.objects.link(cam)
    cam.location = (5, -5, 20)
    
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, -5))
    
    for i in range(len(s1_strokes_kp[0])):
        
        loc = (s1_strokes_kp[0][i][0]*15, s1_strokes_kp[0][i][1]*15, s1_strokes_kp[0][i][1]*15)
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=loc)
        

    ## Get references to all cubes
    
    bpy.data.objects['Cube'].select_set(True)
    for i in range(1, len(s1_strokes_kp[0])):
        bpy.data.objects[f'Cube.{str(("%03d"% i))}'].select_set(True)
    
    cubes = bpy.context.selected_objects
    print(cubes)
    

    ## Insert keyframe at frame one
    
    start_frame = 1
    for i in range(0, len(s1_strokes_kp[0])):
        cubes[i].keyframe_insert("location", frame=start_frame)
    

    ## Change the location of the cubes
    
    for i in range(len(s1_strokes_kp)):
        for j in range(0, len(s1_strokes_kp[0])):
            changeCubeLocation(cubes[j], s1_strokes_kp[i][j][0]*15, s1_strokes_kp[i][j][1]*15, s1_strokes_kp[i][j][2]*15, i)