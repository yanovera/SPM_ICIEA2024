# Run in blender after loading the animation model
"""
This script animates a Blender model of a Spherical Parallel Manipulator (SPM) using data from a single evaluation log file. It reads the log file, extracts the necessary angles, and applies these to the Blender model to create an animation.

Script Overview:
1. Imports and Setup:
   - Imports necessary libraries like math, bpy, csv, and numpy.
   - Defines file paths, initial angles, and animation settings.

2. Classes:
   - Arm: Manages the rotation and keyframe insertion for the manipulator's arms.
   - Platform: Manages the orientation and keyframe insertion for the platform.
   - LaserObject: Manages the visibility of laser objects in the scene.

3. Functions:
   - view3d_fullscreen: Toggles fullscreen mode in Blender's 3D view.
   - stop_playback: Stops the animation playback at the end of the scene.
   - read_angles: Reads angles from the log file and stores them in lists.
   - get_target_position: Computes the target position using rotation matrices.
   - rotate_system: Applies the read angles to the Blender model and inserts keyframes.

4. Main Function:
   - Initializes the arms and platform.
   - Reads the log file and extracts angles.
   - Applies the angles to the Blender model and creates keyframes.
   - Optionally plays and saves the animation.

How to Use:
1. Load the Blender model and run this script in Blender.
2. Ensure the log file path and name are correctly set in the script.
3. Set the desired animation settings (e.g., playAnimation, saveAnimation).
4. Run the script to generate the animation.
"""

import math
import bpy
import csv
import numpy as np

logFilePath = '/home/aviramy/PycharmProjects/SPM/logs/'
logFileName = 'logfile.txt'

movieFilePath = '//'
movieFileName = 'movie'

initialBotArmsAngle = -65.33
initialTopArmsAngle = 99.59

playAnimation = True
saveAnimation = False

scopeLaserOn = True
targetLaserOn = False

syncFrames = False


class Arm:
    def __init__(self, armature, bone):
        self.object = bpy.data.objects[armature]
        self.pbone = self.object.pose.bones[bone]
        self.pbone.rotation_mode = 'XYZ'

    def rotate(self, axis, angle):
        self.pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
        
    def set_rotation(self, angle):
        self.pbone.rotation_euler = (0, 0, 0)
        self.rotate('Y', angle)
        
    def insert_keyframe(self, frame):
        self.pbone.keyframe_insert(data_path="rotation_euler" ,frame=frame)


class Platform:
    def __init__(self, armature, bone):
        self.object = bpy.data.objects[armature]
        self.pbone = self.object.pose.bones[bone]
        self.pbone.rotation_mode = 'XYZ'

    def rotate(self, axis, angle):
        self.pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
        
    def set_orientation(self, psi, theta, phi):
        self.pbone.rotation_euler = (0, 0, 0)
        self.rotate('Y', psi)
        self.rotate('Z', theta)
        self.rotate('X', -phi)
        
    def insert_keyframe(self, frame):
        self.pbone.keyframe_insert(data_path="rotation_euler" ,frame=frame)


class LaserObject:
    def __init__(self, laserObject):
        self.object = bpy.data.objects[laserObject]
        
    def enable(self):
        self.object.hide_viewport = False
        self.object.hide_render = False
        
    def disable(self):
        self.object.hide_viewport = True
        self.object.hide_render = True          

        
# Press CTRL + ALT + SPACE to leave fullscreen mode 
def view3d_fullscreen(dummy):
    bpy.app.handlers.depsgraph_update_post.remove(view3d_fullscreen)
    context = bpy.context.copy()

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED'
            context['area'] = area
            # bpy.ops.screen.screen_full_area(context, use_hide_panels=True)
            break


def stop_playback(scene):
    if scene.frame_current == bpy.data.scenes[0].frame_end:
        bpy.ops.screen.animation_cancel(restore_frame=False)


def read_angles(botArmsAngles, topArmsAngles, platformAngles, targetAngles, filename, rowsPerSecond=1000, framesPerSecond=24, maxLines=5000, syncFrames=False):
    rowsCounter = 0
    skipRows = int(rowsPerSecond/framesPerSecond)
    try:
        with open(filename) as file:
            reader = csv.reader(file)
            next(reader)                # skip header
            next(reader)                # skip header
            for row in reader:
                if rowsCounter == maxLines:
                    break
                if (not syncFrames) or (rowsCounter % skipRows) == 0:
                    botArmsAngles.append([float(row[0]), float(row[1]), float(row[2])])
                    topArmsAngles.append([float(row[3]), float(row[4]), float(row[5])])
                    platformAngles.append([float(row[6]), float(row[7]), float(row[8])])
                    targetVector = [float(row[18]), float(row[19]), float(row[20])]
                    targetAngles.append(np.rad2deg(get_target_position(lookv_goal=targetVector)))
                rowsCounter = rowsCounter + 1
            
    except EnvironmentError:
        print('read error')
    return len(platformAngles)


def get_target_position(lookv_goal, start_roll=0, start_pitch=0, start_yaw=0):
    """Compute target position using rotation matrices."""
    initial_orientation = Q321(start_roll, start_pitch, start_yaw)
    lookv = np.dot(initial_orientation, np.array([0, 0, 1]))

    R = rotation_matrix_from_vectors(lookv, lookv_goal)
    final_orientation = np.matmul(R, initial_orientation)

    phi, theta, psi = euler_from_matrix(final_orientation)
    return phi, theta, psi


def rotation_matrix_from_vectors(a, b):
    """Compute the rotation matrix that rotates vector a to align with vector b."""
    v = np.cross(a, b)
    c = np.dot(a, b)
    I = np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = I + vx + np.matmul(vx, vx) * (1 / (1 + c))
    return R


def euler_from_matrix(matrix):
    """Extract Euler angles from a rotation matrix."""
    pitch = np.arcsin(-matrix[2, 0])
    if np.cos(pitch) != 0:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
    return roll, pitch, yaw


def Q321(x1, x2, x3):
    c1 = np.cos(x1)
    s1 = np.sin(x1)
    c2 = np.cos(x2)
    s2 = np.sin(x2)
    c3 = np.cos(x3)
    s3 = np.sin(x3)

    d1m = np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]])
    d2m = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    d3m = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    y = np.matmul(np.matmul(d3m, d2m), d1m)
    return y


def rotate_system(botArm1, botArm2, botArm3, topArm1, topArm2, topArm3, platform, targetPlatform, botArmsAngles, topArmsAngles, platformAngles, targetAngles, frame):
    botArm1.set_rotation(botArmsAngles[0] - initialBotArmsAngle)
    botArm1.insert_keyframe(frame)
    botArm2.set_rotation(botArmsAngles[1] - initialBotArmsAngle)
    botArm2.insert_keyframe(frame)
    botArm3.set_rotation(botArmsAngles[2] - initialBotArmsAngle)
    botArm3.insert_keyframe(frame)
        
    topArm1.set_rotation(topArmsAngles[0] - initialTopArmsAngle)
    topArm1.insert_keyframe(frame)
    topArm2.set_rotation(topArmsAngles[1] - initialTopArmsAngle)
    topArm2.insert_keyframe(frame)
    topArm3.set_rotation(topArmsAngles[2] - initialTopArmsAngle)
    topArm3.insert_keyframe(frame)
        
    platform.set_orientation(platformAngles[2], platformAngles[1], platformAngles[0])
    platform.insert_keyframe(frame)
    
    targetPlatform.set_orientation(targetAngles[2], targetAngles[1], targetAngles[0])
    targetPlatform.insert_keyframe(frame)


def main():
    botArm1 = Arm('Armature', 'BotArm1')
    botArm2 = Arm('Armature', 'BotArm2')
    botArm3 = Arm('Armature', 'BotArm3')
    topArm1 = Arm('Armature', 'TopArm1')
    topArm2 = Arm('Armature', 'TopArm2')
    topArm3 = Arm('Armature', 'TopArm3')
    platform = Platform('Armature', 'Platform')
    
    bpy.context.scene.frame_set(0)
    platform.insert_keyframe(0) #  save platform's initial position
    
    objectToSelect = bpy.data.objects["Armature"]
    objectToSelect.select_set(True)    
    bpy.context.view_layer.objects.active = objectToSelect
    bpy.ops.object.mode_set(mode='POSE')    

    botArmsAngles = []
    topArmsAngles = []
    platformAngles = []
    targetAngles = []
    
    logFile = logFilePath + logFileName
     
    numActions = read_angles(botArmsAngles, topArmsAngles, platformAngles, targetAngles, logFile, syncFrames=syncFrames)

    if playAnimation:
        bpy.app.handlers.depsgraph_update_post.append(view3d_fullscreen)
        
    laser = LaserObject('Laser')
    targetLaser = LaserObject('TargetLaser')
    if scopeLaserOn:
        laser.enable()
    else:
        laser.disable()
    if targetLaserOn:
        targetLaser.enable()
    else:
        targetLaser.disable()

    targetPlatform = Platform('Armature', 'Target')
    targetPlatform.insert_keyframe(0)

    frame = 1
    for action in range(0, numActions):
        rotate_system(botArm1, botArm2, botArm3, topArm1, topArm2, topArm3, platform, targetPlatform,
                      botArmsAngles[action], topArmsAngles[action], platformAngles[action], targetAngles[action], frame) 
        frame = frame + 1

    bpy.data.scenes[0].frame_start = 1    
    bpy.data.scenes[0].frame_end = frame
        
    if playAnimation:    
        # add one of these functions to frame_change_pre handler:
        bpy.app.handlers.frame_change_pre.append(stop_playback)
        
        #starting animation
        bpy.ops.screen.animation_play()
        bpy.app.handlers.depsgraph_update_post.append(view3d_fullscreen)
        bpy.context.window.workspace = bpy.data.workspaces['Animation'] 
        
    if saveAnimation:
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.filepath = movieFilePath + movieFileName
        bpy.ops.render.render('INVOKE_DEFAULT', animation=True)
   

if __name__ == '__main__':
    main()
    

