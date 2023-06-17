# 2023.0618 @Brian
# Simple animation scripts.

import bpy

def changeCubeLocation(x, y, z, frame):
    
    # Change the location of the cube
    cube.location.x = x
    cube.location.y = y
    cube.location.z = z

    # Insert keyframe at the last frame
    end_frame = frame
    cube.keyframe_insert("location", frame=end_frame)
    
    
if __name__ == "__main__":

    ## Add a cube into the scene
    
    bpy.ops.mesh.primitive_cube_add()

    ## Get a reference to the currently active object
    
    cube = bpy.context.active_object
    print(cube)

    ## Insert keyframe at frame one
    start_frame = 1
    cube.keyframe_insert("location", frame=start_frame)

    ## Change the location of the cube
    trajectory = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    for i in range(len(trajectory)):
        changeCubeLocation(trajectory[i][0], trajectory[i][1], trajectory[i][2], 50*i)