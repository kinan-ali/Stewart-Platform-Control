# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 22:50:14 2024

@author: Kinan ALI
Contact: kinan77ali@gmail.com
"""

import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk  # PIL library for image processing
import serial
import numpy as np
from collections import deque
import math
import time
import threading
import pandas as pd
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import queue
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import itertools
import tensorflow as tf
from keras.optimizers import Adam
import os

#Serial port configuration with Motors Control UNIT (AtMega1280)
port = 'COM10'
baud_rate = 57600
video_cap_source ="http://192.168.161.238:8080/video"
Camera_Calibration_file = "./Stewart_Platform_Control_Kinan/mobCam/mobCAM_720x480.yml"


save_data_directory = './ '

#model&scalers
model123_directory = './Stewart_Platform_Control_Kinan/Neural_Nework/th_Gaussian_div123.h5'
scalers123_directory = './Stewart_Platform_Control_Kinan/Neural_Nework/scalers_Gaussian_div123.pkl'

#Current Sensor port and baud rate
Current_sensor_port='COM6'
Current_sensor_baud_rate = 9600





#                            ███████  ███████  ██████   ██   █████   ██      
#                            ██       ██       ██   ██  ██  ██   ██  ██      
#                            ███████  █████    ██████   ██  ███████  ██      
#                                 ██  ██       ██   ██  ██  ██   ██  ██      
#                            ███████  ███████  ██   ██  ██  ██   ██  ███████ 
filename_global = ''
b_angles = np.array([35,35,35,35,35,35])
angles_global = np.array([35,35,35,35,35,35])
angles_previous_global = np.array([35,35,35,35,35,35])
angles_previous_previous_global = np.array([35,35,35,35,35,35])


th_dir = np.where((angles_global - angles_previous_global > 0) & (angles_previous_global - angles_previous_previous_global <= 0), 1, 
                  np.where((angles_global - angles_previous_global < 0) & (angles_previous_global - angles_previous_previous_global >= 0), -1, 0))

#calculate high and low bytes with boundaries
def calculate_high_low(dec):
    high_byte = dec // 256
    low_byte = dec % 256

    if low_byte > 255:
        low_byte = 255
    if high_byte > 150:
        high_byte = 150
    if high_byte < 144:
        high_byte = 144

    return high_byte, low_byte

#Convert to unsigned char (0-255)
def to_unsigned_char(value):
    return value & 0xFF

def send_data(angles):
    global angles_global, angles_previous_global, angles_previous_previous_global
    global dth, dth_prev, th_dir
    global b_angles
    #Angles boundaries (it's important to be in the safe side)

    
    th1 = angles[0]
    th2 = angles[1]
    th3 = angles[2]
    th4 = angles[3]
    th5 = angles[4]
    th6 = angles[5]
    b_angles = np.array([th1,th2,th3,th4,th5,th6])
    if ibacklash_adaptation_flag is True: 
        th1 = ibacklash1.apply(th1)
        th2 = ibacklash2.apply(th2)
        th3 = ibacklash3.apply(th3)
        th4 = ibacklash4.apply(th4)
        th5 = ibacklash5.apply(th5)
        th6 = ibacklash6.apply(th6)

    if backlash_enable.get() is True:

        backlash1.set_backlash_width(float(entry_backlash1.get()))
        backlash2.set_backlash_width(float(entry_backlash2.get()))
        backlash3.set_backlash_width(float(entry_backlash3.get()))
        backlash4.set_backlash_width(float(entry_backlash4.get()))
        backlash5.set_backlash_width(float(entry_backlash5.get()))
        backlash6.set_backlash_width(float(entry_backlash6.get()))
        th1 = backlash1.apply(th1)
        th2 = backlash2.apply(th2)
        th3 = backlash3.apply(th3)
        th4 = backlash4.apply(th4)
        th5 = backlash5.apply(th5)
        th6 = backlash6.apply(th6)

        b_angles = np.array([th1,th2,th3,th4,th5,th6])

        print (f"angles after backlash:  {np.round([th1,th2,th3,th4,th5,th6],1)}")

    th_angles = [th1,th2,th3,th4,th5,th6]
    for angle in th_angles:
        if angle > 85 or angle < -25:
            print("Data not sent: angle out of bounds")
            return
    
    #Linear transformation (that was done by experiement and using a procractor so the error can be up to 1 degree)
    th1 = 1.3097 * th1 - 13.278
    th2 = 1.2873 * th2 - 13.808
    th3 = 1.2945 * th3 + 0.4823
    th4 = 1.3061 * th4 - 12.402
    th5 = 1.3175 * th5 - 11.306
    th6 = 1.2915 * th6 - 22.222

    angles_degrees = [th1, th2, th3, th4, th5, th6]

    #Conversion formula for each motor angle (it's neccessary for the Atemega code written by Agiad kasem)
    dec1 = 75.0 / 7 * th1 + 13500.0 / 7 + 35536 - 104
    dec2 = -75.0 / 7 * th2 + 7500.0 / 7 + 37036 + 36
    dec3 = 75.0 / 7 * th3 + 13500.0 / 7 + 35536 - 104
    dec4 = -75.0 / 7 * th4 + 7500.0 / 7 + 37036 + 36
    dec5 = 75.0 / 7 * th5 + 13500.0 / 7 + 35536 - 104
    dec6 = -75.0 / 7 * th6 + 7500.0 / 7 + 37036 + 36

    #to integers
    dec_values = [int(dec) for dec in [dec1, dec2, dec3, dec4, dec5, dec6]]

    #calculate high and low bytes for each angle
    high_low_bytes = [calculate_high_low(dec) for dec in dec_values]

    #flatten the list of tuples and convert to unsigned char
    flattened_bytes = [to_unsigned_char(byte) for high_low in high_low_bytes for byte in high_low]

    # Calculate CRC (xor operation of all teh angles) (the crc should be the last byte in the message)
    crc = 0
    for byte in flattened_bytes:
        crc ^= byte

    #convert 'C' to its ASCII value (the letter 'C' or 67 in ASSCI is the head of the message)
    data_to_send = [ord('C')] + flattened_bytes + [crc]

    byte_array_to_send = bytearray(data_to_send)

    # Print the data to be sent
    #print("Data to be sent:", list(byte_array_to_send))

    #sending data via serial port
    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            ser.write(byte_array_to_send)
            #update the global angles variables
            angles_previous_previous_global = angles_previous_global
            angles_previous_global = angles_global
            angles_global = angles
            dth = angles_global - angles_previous_global
            dth_prev = angles_previous_global - angles_previous_previous_global
            th_dir = np.where((angles_global - angles_previous_global > 0) & (angles_previous_global - angles_previous_previous_global <= 0), 1, 
                            np.where((angles_global - angles_previous_global < 0) & (angles_previous_global - angles_previous_previous_global >= 0), -1, 0))

            print (f"sending angles : {np.round(angles_global,1)}")
            print (f'               th direction:  {th_dir}')
        #print("Sent: the data that has been sent")
    except Exception as e:
        print(f"Data not sent: {e}")


#   ______                           __ 
#  / ____/_  _______________  ____  / /_
# / /   / / / / ___/ ___/ _ \/ __ \/ __/
#/ /___/ /_/ / /  / /  /  __/ / / / /_  
#\____/\__,_/_/  /_/   \___/_/ /_/\__/ 

class CurrentSensor:
    global trajectory_step_global
    def __init__(self):
        current_port = Current_sensor_port
        current_baud_rate = Current_sensor_baud_rate
        self.current_value = 0.0
        self.running = True
        self.current_readings = []  # List to store current readings
        
        try:
            self.ser = serial.Serial(current_port, current_baud_rate)
        except serial.SerialException as e:
            print(f"Failed to connect to {current_port}: {e}")
            self.ser = None
        
        if self.ser is not None:
            self.thread = threading.Thread(target=self.update_current_value)
            self.thread.start()
        else:
            self.thread = None

    def update_current_value(self):
        global trajectory_step_global
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                self.current_value = float(line)
                self.current_readings.append(self.current_value)
                
                #limit the size of the list to avoid excessive memory usage
                if len(self.current_readings) > 50:
                    self.current_readings.pop(0)
                    
                if self.get_bigness_of_average_current() > 5.5:
                    trajectory_step_global = 0.1
                elif self.get_bigness_of_average_current() < 5:
                    trajectory_step_global = 0.2
            except Exception as e:
                print(f"Error reading from serial port: {e}")

    def get_current_value(self):
        return self.current_value

    def get_average_current(self):
        if self.current_readings:
            return sum(self.current_readings) / len(self.current_readings)
        else:
            return 0.0

    def get_bigness_of_average_current(self):
        avg_current = self.get_average_current()
        if avg_current >= 10:
            return 10
        elif avg_current >= 9:
            return 9
        elif avg_current >= 8:
            return 8
        elif avg_current >= 7:
            return 7
        elif avg_current >= 6:
            return 6
        elif avg_current >= 5:
            return 5
        elif avg_current >= 4:
            return 4
        elif avg_current >= 3:
            return 3
        elif avg_current >= 2:
            return 2
        elif avg_current >= 1:
            return 1
        else:
            return 0

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.ser is not None:
            self.ser.close()

current_measurements_global = CurrentSensor()


#██████   █████   ██████ ██   ██ ██       █████  ███████ ██   ██ 
#██   ██ ██   ██ ██      ██  ██  ██      ██   ██ ██      ██   ██ 
#██████  ███████ ██      █████   ██      ███████ ███████ ███████ 
#██   ██ ██   ██ ██      ██  ██  ██      ██   ██      ██ ██   ██ 
#██████  ██   ██  ██████ ██   ██ ███████ ██   ██ ███████ ██   ██ 

class BacklashHysteresis:
    def __init__(self, backlash_width):
        self.backlash_width = backlash_width
        self.last_output = 0

    def set_backlash_width(self,new_width):
        self.backlash_width = new_width

    def get_backlash_width(self):
        return self.backlash_width

    def apply(self, current_input):
        if current_input > self.last_output + self.backlash_width:
            self.last_output = current_input - self.backlash_width
        elif current_input < self.last_output - self.backlash_width:
            self.last_output = current_input + self.backlash_width
        # Else, the last_output remains unchanged
        return self.last_output

backlash1 = BacklashHysteresis(backlash_width=0)
backlash2 = BacklashHysteresis(backlash_width=0)
backlash3 = BacklashHysteresis(backlash_width=0)
backlash4 = BacklashHysteresis(backlash_width=0)
backlash5 = BacklashHysteresis(backlash_width=0)
backlash6 = BacklashHysteresis(backlash_width=0)
backlash_enable = False


class IBacklash:
    def __init__(self, backlash_width):
        self.backlash_width = backlash_width
        self.last_output = 0
        self.last_input = 0

    def set_backlash_width(self, new_width):
        self.backlash_width = new_width

    def get_backlash_width(self): 
        return self.backlash_width

    def apply(self, current_input):
        if current_input > self.last_input:
            self.last_output = current_input + self.backlash_width
        elif current_input < self.last_input:
            self.last_output = current_input - self.backlash_width
        else:
            self.last_output = current_input

        self.last_input = current_input
        return self.last_output
ibacklash1 = IBacklash(backlash_width=0)
ibacklash2 = IBacklash(backlash_width=0)
ibacklash3 = IBacklash(backlash_width=0)
ibacklash4 = IBacklash(backlash_width=0)
ibacklash5 = IBacklash(backlash_width=0)
ibacklash6 = IBacklash(backlash_width=0)

ibacklash_adaptation_flag = False

def BacklashDetector(compensated_input, backlashed_output, tolerance=0.5):
    difference = compensated_input - backlashed_output
    if difference > tolerance:
        return 1
    elif difference < -tolerance:
        return -1
    # No significant difference detected, return 0
    else:
        return 0

#██   ██████   ███    ███     
#██  ██        ████  ████ 
#██  ██   ███  ██ ████ ██ 
#██  ██    ██  ██  ██  ██ 
#██   ██████   ██      ██ 


#IGM configuration: 
b = 26
d = 20.3694
bb = 30
dd = 6
LL1 = 10
LL2 = 25.375

# Define rotation matrices
R1 = np.array([
    [np.sqrt(3) / 2, 0, -0.5, (np.sqrt(3) / 6) * (2 * b + d)],
    [-0.5, 0, -np.sqrt(3) / 2, d / 2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

R2 = np.array([
    [-np.sqrt(3) / 2, 0, 0.5, -(np.sqrt(3) / 6) * (b - d)],
    [0.5, 0, np.sqrt(3) / 2, (b + d) / 2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

R3 = np.array([
    [0, 0, 1, -(np.sqrt(3) / 6) * (b + 2 * d)],
    [1, 0, 0, b / 2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

R4 = np.array([
    [0, 0, -1, -(np.sqrt(3) / 6) * (b + 2 * d)],
    [-1, 0, 0, -b / 2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

R5 = np.array([
    [-np.sqrt(3) / 2, 0, -0.5, -(np.sqrt(3) / 6) * (b - d)],
    [-0.5, 0, np.sqrt(3) / 2, -(b + d) / 2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

R6 = np.array([
    [np.sqrt(3) / 2, 0, 0.5, (np.sqrt(3) / 6) * (2 * b + d)],
    [0.5, 0, -np.sqrt(3) / 2, -d / 2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

# Define the A and C points
A = np.array([
    [(np.sqrt(3) / 6) * (2 * b + d), d / 2, 0],
    [-(np.sqrt(3) / 6) * (b - d), (b + d) / 2, 0],
    [-(np.sqrt(3) / 6) * (2 * d + b), b / 2, 0],
    [-(np.sqrt(3) / 6) * (2 * d + b), -b / 2, 0],
    [-(np.sqrt(3) / 6) * (b - d), -(b + d) / 2, 0],
    [(np.sqrt(3) / 6) * (2 * b + d), -d / 2, 0]
]).T

Cp = np.array([
    [(np.sqrt(3) / 6) * (2 * dd + bb), bb / 2, 0],
    [(np.sqrt(3) / 6) * (bb - dd), (bb + dd) / 2, 0],
    [-(np.sqrt(3) / 6) * (2 * bb + dd), dd / 2, 0],
    [-(np.sqrt(3) / 6) * (2 * bb + dd), -dd / 2, 0],
    [(np.sqrt(3) / 6) * (bb - dd), -(bb + dd) / 2, 0],
    [(np.sqrt(3) / 6) * (2 * dd + bb), -bb / 2, 0]
]).T

def IGM(x):
    # Extract the inputs
    xx = x[0]
    y = x[1]
    z = x[2]
    psi = x[3] * (np.pi / 180)
    theta = x[4] * (np.pi / 180)
    phi = x[5] * (np.pi / 180)


    # Define the rotation matrix R
    R = np.array([
        [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)],
        [-np.sin(theta), np.cos(theta) * np.sin(psi), np.cos(theta) * np.cos(psi)]
    ])

    # Calculate C points
    C = np.outer(np.array([xx, y, z]), np.ones(6)) + R @ Cp

    # Calculate lengths L
    L = np.sqrt(np.sum((C - A) ** 2, axis=0))

    # Calculate AC vectors
    AC = np.vstack([C - A, np.zeros((1, 6))])

    # Calculate s vectors
    s = [np.linalg.inv(Ri) @ AC[:, i] for Ri, i in zip([R1, R2, R3, R4, R5, R6], range(6))]

    # Calculate p values
    p = [(L[i] ** 2 + LL1 ** 2 - LL2 ** 2) / (2 * LL1) for i in range(6)]

    # Calculate k values
    k = [s[i][0] ** 2 + s[i][1] ** 2 - p[i] ** 2 for i in range(6)]

    # Calculate theta values
    theta = [np.arctan2(
        (p[i] * s[i][1] - s[i][0] * np.sqrt(k[i])) / (s[i][0] ** 2 + s[i][1] ** 2),
        (p[i] * s[i][0] + s[i][1] * np.sqrt(k[i])) / (s[i][0] ** 2 + s[i][1] ** 2)
    ) * (180 / np.pi) for i in range(6)]

    return np.array(theta)


#               ██████   █████   ███    ███  ███████  ██████    █████  
#              ██       ██   ██  ████  ████  ██       ██   ██  ██   ██ 
#              ██       ███████  ██ ████ ██  █████    ██████   ███████ 
#              ██       ██   ██  ██  ██  ██  ██       ██   ██  ██   ██ 
#               ██████  ██   ██  ██      ██  ███████  ██   ██  ██   ██ 
                                                   

base_aruco_num_samples = 25

#Aruco Configuration: 
marker_length_end_effector = 9.4  # cm
marker_length_base = 9.4 #8 # cm

end_effector_marker_id = 0
base_reference_marker_id = 0

aruco_ids = [end_effector_marker_id,
             base_reference_marker_id]

#Camera Config: 
ARUCO_type = cv2.aruco.DICT_4X4_250
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_type)
board_size = (4, 3)  #(Number of markers in x, Number of markers in y)
marker_length = 5.5  #marker side length in cm
marker_separation = 1.5  
board = cv2.aruco.GridBoard(board_size, marker_length, marker_separation, dictionary)


fs = cv2.FileStorage(Camera_Calibration_file, cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("K")
camera_matrix = camera_matrix.mat()
dist_coeffs = fs.getNode("D")
dist_coeffs = dist_coeffs.mat()

class CameraCapture:
    def __init__(self, video_source=video_cap_source):
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.vid.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.vid.release()

def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec)) 
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    # Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec

def calculate_rotation(rvec, tvec):
    rmat = cv2.Rodrigues(rvec)[0]
    P = np.concatenate((rmat, np.reshape(tvec, (rmat.shape[0], 1))), axis=1)
    eul = -cv2.decomposeProjectionMatrix(P)[6]
    
    yaw = eul[1, 0] 
    pitch = eul[0, 0]
    roll = eul[2, 0]
    
    return pitch, yaw, roll

def convert_Rodrigues2euler(rvecs):
  angle = np.sqrt(rvecs[0]*rvecs[0] + rvecs[1]*rvecs[1] + rvecs[2]*rvecs[2])
  print (f"angle (degrees) =                                {angle*180/math.pi}")
  q = [         math.cos (angle/2),
                rvecs[0]* math.sin(angle/2),
                rvecs[1]* math.sin(angle/2),
                rvecs[2]* math.sin(angle/2)]
  psi = math.atan2( -2*(q[2]*q[3] - q[0]*q[1]) , q[0]*q[0] - q[1]*q[1]- q[2]*q[2] + q[3]*q[3] )
  theta = math.asin( 2*(q[1]*q[3] + q[0]*q[2]) )
  phi = math.atan2( 2*(-q[1]*q[2] + q[0]*q[3]) , q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3] )  

  psi = psi*180/math.pi   
  theta = theta*180/math.pi   
  phi = phi*180/math.pi   
  return psi, theta, phi

def Rod2Euler(rvecs):
    angle = np.sqrt(rvecs[0]*rvecs[0] + rvecs[1]*rvecs[1] + rvecs[2]*rvecs[2])
    q = [         math.cos (angle/2),
                    rvecs[0]* math.sin(angle/2),
                    rvecs[1]* math.sin(angle/2),
                    rvecs[2]* math.sin(angle/2)]
    #roll (x-axis rotation)
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    roll = math.atan2(sinr_cosp, cosr_cosp)

    #pitch (y-axis rotation)
    sinp = math.sqrt(1 + 2 * (q[0] * q[2] - q[1] * q[3]))
    print (1 - 2 * (q[0] * q[2] - q[1] * q[3]))
    cosp = math.sqrt(1 - 2 * (q[0] * q[2] - q[1] * q[3]))
    pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2

    #yaw (z-axis rotation)
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    yaw = math.atan2(siny_cosp, cosy_cosp)

    roll = roll*180/math.pi   
    pitch = pitch*180/math.pi   
    yaw = yaw*180/math.pi   
    return roll,pitch,yaw

def Rod2Euler2(rvecs):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        angle = np.sqrt(rvecs[0]*rvecs[0] + rvecs[1]*rvecs[1] + rvecs[2]*rvecs[2])
        angle = angle[0]
        #print (f"angle (degrees) =                                {np.round(angle*180/math.pi,2)}")
        w=math.cos (angle/2)
        x=rvecs[0]* math.sin(angle/2) / angle  
        y=rvecs[1]* math.sin(angle/2) / angle
        z=rvecs[2]* math.sin(angle/2) /angle

        x=x[0]
        y=y[0]
        z=z[0]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        roll_x = roll_x * 180/math.pi
        pitch_y = pitch_y*180/math.pi
        yaw_z = yaw_z * 180/math.pi

        roll_x = 0 if np.isnan(roll_x) else roll_x
        pitch_y = 0 if np.isnan(pitch_y) else pitch_y
        yaw_z = 0 if np.isnan(yaw_z) else yaw_z

        return roll_x, pitch_y, yaw_z # degrees

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
#the euler angles ( x and z are swapped ).
def Rod2Euler3(rvec) :
    
    R, _ = cv2.Rodrigues(rvec)

    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        roll = math.atan2(R[2,1] , R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else :
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0
 
    roll = roll * 180/math.pi
    pitch = pitch * 180/math.pi
    yaw = yaw * 180/math.pi
    return yaw, roll,pitch

def median_filter(lst):
    #find the median of the angles corresponding to each rvec in the base measurement
    norms = [np.linalg.norm(sublist) for sublist in lst]
    
    median_norm = np.median(norms)
    
    median_sublist_index = norms.index(median_norm)
    median_sublist = lst[median_sublist_index]
    
    return median_sublist

def lp_filter(old_data, measurement, alpha=0.5): 
    filtered_data = alpha*old_data + (1-alpha)*measurement
    return filtered_data

base_Aruco_position_list = deque(maxlen=base_aruco_num_samples)
base_Aruco_rotation_list = deque(maxlen=base_aruco_num_samples)

base_list_full =True

base_Aruco_position_global= np.array([-2.79612512, -1.63965937, 61.68591216])
base_Aruco_rotation_global = np.array([ 0.54977871, -2.80157651,  0.12791043])

end_effector_POSE_base_global = [0,0,19,0,0,0]
end_effector_POSE_previous = [0,0,0,0,0,0]

end_effector_VELOCITY_global = [0,0,0,0,0,0]

def estimate_pose(frame):
    global end_effector_POSE_previous, end_effector_VELOCITY_global
    global end_effector_POSE_base_global
    global base_Aruco_position_global , base_Aruco_rotation_global
    global base_Aruco_list, base_list_full
    #detect markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is not None:
        # Refine detected markers
        corners, ids, rejected, recoveredIdxs = cv2.aruco.refineDetectedMarkers(
            gray, board, corners, ids, rejected, camera_matrix, dist_coeffs
        )
        # Initialize the rotation and translation vectors
        rvec = np.zeros((1, 3), dtype=np.float64)
        tvec = np.zeros((1, 3), dtype=np.float64)
        # Estimate the pose of the board
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
        
        end_effector_position_camera = tvec.flatten()
        end_effector_rotation_camera = rvec.flatten()
        R, _ = cv2.Rodrigues(rvec)                    
        end_effector_position_camera = end_effector_position_camera + np.dot(R, np.array([[13.1], [9.4], [4]])).flatten()   

        #apply rotations:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        tvec = end_effector_position_camera
        # Define 180 degree rotation matrix around the x-axis
        rotation_matrix_x180 = np.array([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]], dtype=np.float64)
        # Multiply the two rotation matrices
        R_rotated = np.dot(rotation_matrix,rotation_matrix_x180)
        #Rotation 55 degree
        rotation_matrix_x55 = np.array([[math.cos(-math.pi*55/180), -math.sin(-math.pi*55/180), 0],
                                        [math.sin(-math.pi*55/180), math.cos(-math.pi*55/180), 0],
                                        [0, 0, 1]], dtype=np.float64)
        
        R_rotated = np.dot(R_rotated,rotation_matrix_x55)

        # Convert the rotated rotation matrix back to a rotation vector
        rvec, _ = cv2.Rodrigues(R_rotated)



        end_effector_position_camera = tvec.flatten()
        end_effector_rotation_camera = rvec.flatten()
        if retval:
            # Draw the markers and the board
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Draw the coordinate axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 10)
            #cv2.putText(frame, f"Translation: {tvec.flatten()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.putText(frame, f"Rotation: {rvec.flatten()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if not base_list_full:
            ref1_marker_position_camera = tvec.flatten()
            ref1_marker_rotation_camera = rvec.flatten()

            base_Aruco_position_list.append(ref1_marker_position_camera)
            base_Aruco_rotation_list.append(ref1_marker_rotation_camera)
            if len(base_Aruco_position_list) == base_aruco_num_samples:
                base_list_full = True
                base_Aruco_position_global = np.mean(base_Aruco_position_list, axis=0)
                base_Aruco_rotation_global = median_filter(base_Aruco_rotation_list)
                print ("                    FINAL REFERENCE DATA        ")
                print (f"base_Aruco_position_global:      {base_Aruco_position_global}")
                print ("/////////////////////////////////////////////////////////////")
                print (f"base_Aruco_rotation_global:      {base_Aruco_rotation_global}")
                print ("                    :)   ")
                #base_Aruco_position_global = np.array([-2.87602449, -1.61541013, 62.03796934])
                #base_Aruco_rotation_global = np.array([ 0.46660631, -2.81750499,  0.12204862])


        if base_list_full and end_effector_position_camera is not None:
            end_effector_rotation_ref1, end_effector_position_ref1 = relativePosition (
                end_effector_rotation_camera, end_effector_position_camera,
                base_Aruco_rotation_global, base_Aruco_position_global
            )

            roll,pitch, yaw  = Rod2Euler2(end_effector_rotation_ref1)

            
            end_effector_POSE_previous =np.array(end_effector_POSE_base_global)
            
            end_effector_POSE_base_global[0] = lp_filter(end_effector_POSE_base_global[0],end_effector_position_ref1[0][0],0.70)#0.85)#0.5)
            end_effector_POSE_base_global[1] = lp_filter(end_effector_POSE_base_global[1],end_effector_position_ref1[1][0],0.70)#0.6)#0.81)
            end_effector_POSE_base_global[2] = lp_filter(end_effector_POSE_base_global[2],end_effector_position_ref1[2][0]+24,0.70)#0.75)#0.89)
            end_effector_POSE_base_global[3] = lp_filter(end_effector_POSE_base_global[3],roll,0.70)#0.75)#0.84)
            end_effector_POSE_base_global[4] = lp_filter(end_effector_POSE_base_global[4],pitch,0.70)#0.73)#0.8)
            end_effector_POSE_base_global[5] = lp_filter(end_effector_POSE_base_global[5],yaw,0.70)#0.7)#0.84)

            
            end_effector_VELOCITY_global = end_effector_POSE_base_global - end_effector_POSE_previous



    return frame


#██    ██  ██████  ██  ██████ ███████ 
#██    ██ ██    ██ ██ ██      ██      
#██    ██ ██    ██ ██ ██      █████   
# ██  ██  ██    ██ ██ ██      ██      
#  ████    ██████  ██  ██████ ███████ 


import pyttsx3
voice_engine = pyttsx3.init()

voice_engine.setProperty('rate', 150)    #speed of speech
voice_engine.setProperty('volume', 1.0)  #volume level (0.0 to 1.0)


# ████████ ██████   █████       ██ ███████  ██████ ████████  ██████  ██████  ██    ██ 
#    ██    ██   ██ ██   ██      ██ ██      ██         ██    ██    ██ ██   ██  ██  ██  
#    ██    ██████  ███████      ██ █████   ██         ██    ██    ██ ██████    ████   
#    ██    ██   ██ ██   ██ ██   ██ ██      ██         ██    ██    ██ ██   ██    ██    
#    ██    ██   ██ ██   ██  █████  ███████  ██████    ██     ██████  ██   ██    ██    
                                                                            

#the equation is from lemniscate of Bernoulli
def infinity_sign(t, radius = 10):
    theta = 2 * math.pi * t/30
    scale = 2 * radius / (3 - math.cos(2*theta))
    x = scale * math.cos(theta)
    y = scale * math.sin(2*theta) / 2
    z = 24
    theta = 0 
    phi = 0 
    psi = 0

    return x,y,z,theta,phi,psi

#def spiral_trajectory(t):
def p2c(r, phi):
    """Convert from polar to Cartesian coordinates."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def spiral_trajectory(t, separation=2, arc=0.1,trajectory_step = 0.1):
    """Return the Cartesian coordinates of the spiral point and z value at a given time."""
    index = int(t * 10)  # Convert time to index
    num_points = int(90 / trajectory_step)
    r = arc
    b = separation / (2 * np.pi)
    phi = arc / b
    
    # Linear interpolation for z value between 19 and 29
    z_min = 19
    z_max = 29
    z = z_min + (z_max - z_min) * (index / (num_points - 1))
    
    # Iterate to find the point at the given index

    my_index = index
    if index >=450 : 
        my_index = 900 -index
    for i in range(my_index):
        phi += arc / r
        r = b * phi
    
    # Calculate Cartesian coordinates for the given index
    x, y = p2c(r, phi)
    
    w = 2 * math.pi / 90  # Frequency based on the provided step size
    theta = 3 * math.sin(w * t)
    phi_angle = 3 * math.sin(w * t + 2 * math.pi / 3)  # Shifted by 120 degrees
    psi = 3 * math.sin(w * t + 4 * math.pi / 3)        # Shifted by 240 degrees

    return x, y, z, theta, phi_angle, psi
def flower(t, radius=7):
    
    if t <= 15:
        theta = 2 * math.pi * t / 15
        scale = 2 * radius / (3 - math.cos(2 * theta))
        x = scale * math.cos(theta)
        y = scale * math.sin(2 * theta) / 2
    else:
        theta = 2 * math.pi * (t - 15) / 15
        scale = 2 * radius / (3 - math.cos(2 * theta))
        x = scale * math.sin(2 * theta) / 2
        y = scale * math.cos(theta)
    
    if t<30 :
        z = 20 + 4 * t / 15
    elif t<60 :
        z = 28 - 4 * (t-30) / 15 
    elif t<=90: 
        z = 20 + 4 * (t-60) / 15
    else : 
        z = 25

    
    theta = 0 
    phi = 0 
    psi = 0

    return x, y, z, theta, phi, psi

def main_trajectory(t):
    x = 4 * math.sin(t * math.pi / 15)
    y = 4 * math.cos(t * math.pi / 15)
    z = 24.5 +3 * math.sin(t * math.pi / 15)

    psi= 3 * math.sin(t * math.pi / 15)
    phi = -4 * math.sin(t * math.pi / 15)
    theta = -4 * math.cos(t * math.pi / 15)

    return x, y, z, theta, phi, psi

def theta_trajectory(t):
    x = 0
    y = 0
    z = 24

    theta = -25 + 25 * t /15  
    phi = 0
    psi = 0 
    return x, y, z, theta, phi, psi

def phi_trajectory(t):
    x = 0
    y = 0
    z = 24

    theta = 0 
    phi = -25 + 25 * t /15 
    psi = 0 
    return x, y, z, theta, phi, psi

def psi_trajectory(t):
    x = 0
    y = 0
    z = 24

    theta = 0 
    phi = 0
    psi = -25 + 25 * t /15
    return x, y, z, theta, phi, psi

def consecutive_triangles(t):
    x = 0
    y = 0 
    z = 24
    theta = 0 
    phi = 0 
    psi = 0

    if t <= 2 : 
        x = 0
        y = 0 
        z = 24
        theta = 0 
        phi = 0 
        psi = 0
    elif t <= 6 : 
        if t <= 4 : 
            x = 8 * (t - 2) / 2 
        else : 
            x = 8 - 8 * (t - 4) / 2
    elif t <= 10 : 
        if t <= 8 : 
            y = 8 * (t - 6) / 2
        else :
            y = 8 - 8 * (t - 8) / 2
    elif t <= 14 : 
        if t <= 12 : 
            z = 24 + 6 * (t - 10) / 2
        else : 
            z = 24 + 6 - 6 * (t - 12) / 2
    elif t <= 18 : 
        if t <= 16 : 
            theta = 25 * (t - 14) / 2
        else : 
            theta = 25 - 25 * (t - 16) / 2
    elif t <= 22 : 
        if t <= 20 : 
            phi = 25 * (t - 18) / 2
        else : 
            phi = 25 - 25 * (t - 20) / 2
    elif t <= 26 : 
        if t <= 24 : 
            psi = 25 * (t - 22) / 2
        else : 
            psi = 25 - 25 * (t - 24) / 2
    
    return x, y, z, theta, phi, psi

def consecutive_merged_steps(t):
    " the time should be up to 44 "
    # Define initial position and angles
    x = 0
    y = 0
    z = 24
    theta = 0
    phi = 0
    psi = 0

    # Calculate the current section based on time
    section_duration = 30 / 9  # Each section is approximately 3.33 seconds
    section = int(t // section_duration)

    # Define the steps for each section
    if section == 0:
        x = 10  # Step on x-axis to 10
    elif section == 1:
        x = -10  # Step on x-axis to -10
    elif section == 2:
        y = 10  # Step on y-axis to 10
    elif section == 3:
        y = -10  # Step on y-axis to -10
    elif section == 4:
        z = 30  # Step on z-axis to 30
    elif section == 5:
        z = 18  # Step on z-axis to 18
    elif section == 6:
        theta = 25  # Step on theta to 45 degrees
    elif section == 7:
        phi = 25  # Step on phi to 30 degrees
    elif section == 8:
        psi = 25  # Step on psi to 60 degrees

    # Steps at the end of the trajectory
    if t >= 30:
        if t >= 30 + 2:
            x = 5  # Final step on x-axis to 20
        if t >= 30 + 4:
            y = 5  # Final step on y-axis to 20
        if t >= 30 + 6:
            z = 24+3  # Final step on z-axis to 40
        if t >= 30 + 8:
            theta = 5  # Final step on theta to 90 degrees
        if t >= 30 + 10:
            phi = 5  # Final step on phi to 60 degrees
        if t >= 30 + 12:
            psi = 5  # Final step on psi to 120 degrees

    return x, y, z, theta, phi, psi


# ██████  ██████  ███    ██ ████████ ██████   ██████  ██      
#██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██      
#██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██      
#██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██      
#██████   ██████  ██   ████    ██    ██   ██  ██████  ███████ 


error_integral = [0, 0, 0, 0, 0, 0]
#closed loop
def closed_loop_control(desired_pos): 
    global angles_global
    global error_integral
    global end_effector_VELOCITY_global
    
    desired_angles = IGM(desired_pos)
    real_angles = IGM(end_effector_POSE_base_global)
    real_angles = np.nan_to_num(real_angles, nan=0.0)

    error = desired_angles - real_angles

    if all(e < 3 for e in error):
        error_integral = error_integral + error
        print (f"error:   {np.round(error,1)},     error_integral:  {np.round(error_integral,1)}")
    if any(abs(e) > 5 for e in error_integral):
        error_integral = np.zeros_like(error_integral)
    Kp, Kd, Ki = 0.22, 0.0, 0.0

    error_integral = np.array(error_integral)
    angles = angles_global + Kp*error + Kd*end_effector_VELOCITY_global + Ki*error_integral

    return angles



custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError()  # or tf.keras.metrics.mean_squared_error
}


# Model 123
try:
    with open(scalers123_directory, 'rb') as f:
        scalers123 = pickle.load(f)
        s_x123 = scalers123['s_x']
        s_y123 = scalers123['s_y']    
    print("Scalers loaded successfully.")
except FileNotFoundError:
    print("Error: The scaler file 'scalers_RNN_model8.pkl' was not found.")
except Exception as e:
    print(f"An error occurred while loading the scalers: {e}")

custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError()  # or tf.keras.metrics.mean_squared_error
}

# Load the model
try:
    model123 = load_model(model123_directory, custom_objects=custom_objects)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: The model file 'th_Gaussian_div123.h5' was not found.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")


def model123_predict(desired_location):
    delta_pose = np.array(desired_location) - np.array(end_effector_POSE_base_global)
    
    input_data = np.concatenate([end_effector_POSE_base_global, delta_pose])
    
    input_scaled = s_x123.transform(input_data.reshape(1, -1))
    
    prediction_scaled = model123.predict(input_scaled)
    
    delta_angles = s_y123.inverse_transform(prediction_scaled)

    final_angles = np.array(angles_global) + 0.5*delta_angles.flatten()
 
    return final_angles








#██████   █████  ████████  █████       ██████  ██████  ██      ██      ███████  ██████ ████████ 
#██   ██ ██   ██    ██    ██   ██     ██      ██    ██ ██      ██      ██      ██         ██    
#██   ██ ███████    ██    ███████     ██      ██    ██ ██      ██      █████   ██         ██    
#██   ██ ██   ██    ██    ██   ██     ██      ██    ██ ██      ██      ██      ██         ██    
#██████  ██   ██    ██    ██   ██      ██████  ██████  ███████ ███████ ███████  ██████    ██  

desired_pose_global = []
desired_pose_previous_global = []
backlash_signal = []
compensated_signal = []
estimated_widths = []  # To store estimated backlash widths over time

trajectory_step_global = 0.1
def collect_data(selected_Trajectory):
    global stop_flag
    global desired_pose_global, desired_pose_previous_global, angles_global, angles_previous_global, th_dir
    global end_effector_POSE_base_global
    global trajectory_step_global
    global ibacklash_adaptation_flag, backlash_enable

    print (f"The Trajectory is:    {selected_Trajectory}")
    control = choose_control_combobox.get()
    print (f"The control is: {control}")
    data = []
    online_data = []
    start_time = time.time()
    



    t = 0    
    while t < 90: 
        if stop_flag:
            break
        if selected_Trajectory == "main Trajectory":
            x, y, z, theta, phi, psi = main_trajectory(t)
        elif selected_Trajectory == "inf Trajectory": 
            x, y, z, theta, phi, psi = infinity_sign(t, 7)
        elif selected_Trajectory == "Consecutive Triangles":
            x, y, z, theta, phi, psi = consecutive_triangles(t)
        elif selected_Trajectory == "Flower Trajectory":
            x, y, z, theta, phi, psi = flower(t,6)
        elif selected_Trajectory == "consecutive merged steps": 
            x, y, z, theta, phi, psi = consecutive_merged_steps(t) # t = 42 sec to complete this trajectory
        elif selected_Trajectory =="Archemides Spiral":
            x, y, z, theta, phi, psi =spiral_trajectory(t)
        else:
            x=0
            y=0
            z=24
            theta = 0 
            phi = 0 
            psi = 0
        desired_pose_previous_global = desired_pose_global
        desired_pose_global[:] = [x, y, z, theta, phi, psi] 
        print (f"Trajectory step:               {trajectory_step_global}")
        
        uncertainty_detected = False
        if t>1: 
            uncertainty_detected = True
        if control == "open loop":
            angles = IGM([x, y, z, theta, phi, psi])
            send_data(angles)
            time.sleep(0.012)  #adjust the delay as needed
        elif control == "closed loop" : 
            angles = closed_loop_control(desired_pose_global)
            send_data(angles)
            time.sleep(0.12) 
        elif control == "Data Driven":
            angles = model123_predict(desired_pose_global)
            send_data(angles)

            time.sleep(0.12)

            send_data(angles)
            time.sleep(0.09)
        elif control == "Adaptive Data Driven":
            angles = model123_predict(desired_pose_global)

            if uncertainty_detected:
                estimated_angles = IGM(end_effector_POSE_base_global)
                if abs(estimated_angles[0] - angles_global[0])>5: 
                    ibacklash1.set_backlash_width(ibacklash1.get_backlash_width() + 0.5)
                if abs(estimated_angles[1] - angles_global[1])>5: 
                    ibacklash2.set_backlash_width(ibacklash2.get_backlash_width() + 0.5)
                if abs(estimated_angles[2] - angles_global[2])>5: 
                    ibacklash3.set_backlash_width(ibacklash3.get_backlash_width() + 0.5)
                if abs(estimated_angles[3] - angles_global[3])>5: 
                    ibacklash4.set_backlash_width(ibacklash4.get_backlash_width() + 0.5)
                if abs(estimated_angles[4] - angles_global[4])>5: 
                    ibacklash5.set_backlash_width(ibacklash5.get_backlash_width() + 0.5)
                if abs(estimated_angles[5] - angles_global[5])>4.5: 
                    ibacklash6.set_backlash_width(ibacklash6.get_backlash_width() + 0.5)

            send_data(angles)
            time.sleep(0.101)
            time.sleep(0.012)  # Adjust the delay as needed



        end_effector_POSE_base_global = np.array(end_effector_POSE_base_global)
        desired_pose_global = np.array(desired_pose_global)
        angles_global = np.array(angles_global)
        current_measurement = current_measurements_global.get_current_value()
        delta_angles = angles_global - angles_previous_global

        combined_data = np.concatenate((end_effector_POSE_base_global, desired_pose_global, 
                                        angles_global, [current_measurement], th_dir,delta_angles,
                                        b_angles))
        data.append(combined_data)
        t += trajectory_step_global

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The trajectory 1st part took {elapsed_time:.6f} seconds to execute.")
    start_time = time.time()
    print ("ESTIMATED BACKLASH COMPENSATION PARAMETERS:")
    print (ibacklash1.get_backlash_width(),
           ibacklash2.get_backlash_width(),
           ibacklash3.get_backlash_width(),
           ibacklash4.get_backlash_width(),
           ibacklash5.get_backlash_width(),
           ibacklash6.get_backlash_width())
    ibacklash1.set_backlash_width(0)
    ibacklash2.set_backlash_width(0)
    ibacklash3.set_backlash_width(0)
    ibacklash4.set_backlash_width(0)
    ibacklash5.set_backlash_width(0)
    ibacklash6.set_backlash_width(0)
    return data

def save_to_excel(data, selected_trajectory):
    global filename_global
    control = choose_control_combobox.get()
    GUI_file_name = file_name_textBox.get("1.0", "end").strip()
    filename= save_data_directory+ control +' ' + selected_trajectory+ ' ' + GUI_file_name + '.xlsx'
    filename_global = filename
    df = pd.DataFrame(data, columns=['EndEffector_x', 'EndEffector_y', 'EndEffector_z',
                                     'EndEffector_theta', 'EndEffector_phi', 'EndEffector_psi',
                                     'Desired_x', 'Desired_y', 'Desired_z',
                                     'Desired_theta', 'Desired_phi', 'Desired_psi',
                                     'th1', 'th2', 'th3', 'th4', 'th5', 'th6', 'current',
                                     'th1_dir', 'th2_dir', 'th3_dir', 'th4_dir', 'th5_dir', 'th6_dir',
                                     'dth1', 'dth2','dth3', 'dth4', 'dth5', 'dth6',
                                     'bth1', 'bth2', 'bth3', 'bth4', 'bth5', 'bth6'])



    df.to_excel(filename, index=False)
    print ("               ***    TRAJECTORY DATA SAVED    ***")
    print ("               ***                             ***")
    print (f"           File Name: {filename}")
    voice_engine.say("mission accomplished!")
    voice_engine.runAndWait()
    voice_engine.say("Trajectory Data has been saved!")
    voice_engine.runAndWait() 



#                        ██████  ██    ██ ██ 
#                       ██       ██    ██ ██ 
#                       ██   ███ ██    ██ ██ 
#                       ██    ██ ██    ██ ██ 
#                        ██████   ██████  ██ 

desired_pose_global = [0,0,0,24,0,0,0]
stop_flag = False
def follow_main_trajectory_GUI():
    global stop_flag
    selected_Trajectory = choose_trajectory_combobox.get()
    data = collect_data(selected_Trajectory)
    save_to_excel(data, selected_Trajectory)
    global desired_pose_global

def start_trajectory_thread():
    voice_engine.say("Robot!, Ready to engage")
    voice_engine.runAndWait()
    global stop_flag
    stop_flag = False
    trajectory_thread = threading.Thread(target=follow_main_trajectory_GUI)
    trajectory_thread.start()

# Stop function
def stop_trajectory():
    global stop_flag
    stop_flag = True

def calibrate():
    global base_list_full, base_Aruco_position_list, base_Aruco_rotation_list
    base_list_full = False
    base_Aruco_position_list = deque(maxlen=base_aruco_num_samples)
    base_Aruco_rotation_list = deque(maxlen=base_aruco_num_samples)

    voice_engine.say("Camera has been calibrated")
    voice_engine.runAndWait() 



#theta and psi are swapped (fix this later)
def send_data_GUI():
    global angles_global
    global desired_pose_global
    voice_engine.say("at your service")
    voice_engine.runAndWait()
    desired_pose = [
        float(entry_desired_x.get()),
        float(entry_desired_y.get()),
        float(entry_desired_z.get()),
        float(entry_desired_theta.get()),
        float(entry_desired_phi.get()),
        float(entry_desired_psi.get())
    ]
    desired_pose_global = desired_pose
    control = choose_control_combobox.get()
    if control == "open loop":
        angles = IGM(desired_pose)
    elif control == "closed loop":
        angles = closed_loop_control(desired_pose)
    elif control == "Data Driven":
        angles = model123_predict(desired_pose)
    send_data(angles)


#gui appearance and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("Stewart Platform Control")
root.geometry("1200x600")

tabview = ctk.CTkTabview(root,width=1180, height=590)
tabview.place(x=10,y=0)




# Desired pose logic
Desired_Pose_frame = ctk.CTkFrame(root, width=300, height=400, corner_radius=10, fg_color="gray", border_color="blue", border_width=2)
Desired_Pose_frame.place(x=20,y=20)

label_Desired_Pose = ctk.CTkLabel(Desired_Pose_frame, text="Desired Pose", font=('Times New Roman', 24))
label_Desired_Pose.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

desired_pose_entrt_width = 45
desired_pose_pady = 2
desired_pose_entry_font = ('Times New Roman', 16)
desired_pose_label_font = ('Times New Roman', 18)

label_desired_x = ctk.CTkLabel(Desired_Pose_frame, text="xᵣ: ", font=desired_pose_label_font)
label_desired_x.grid(row=1, column=0, pady=desired_pose_pady, padx=(10, 5))

entry_desired_x = ctk.CTkEntry(Desired_Pose_frame, width=desired_pose_entrt_width, font=desired_pose_entry_font)
entry_desired_x.grid(row=2, column=0, pady=desired_pose_pady, padx=(5, 10))

label_desired_y = ctk.CTkLabel(Desired_Pose_frame, text="yᵣ: ", font=desired_pose_label_font)
label_desired_y.grid(row=1, column=1, pady=desired_pose_pady, padx=(10, 5))

entry_desired_y = ctk.CTkEntry(Desired_Pose_frame, width=desired_pose_entrt_width, font=desired_pose_entry_font)
entry_desired_y.grid(row=2, column=1, pady=desired_pose_pady, padx=(5, 10))

label_desired_z = ctk.CTkLabel(Desired_Pose_frame, text="zᵣ: ", font=desired_pose_label_font)
label_desired_z.grid(row=1, column=2, pady=desired_pose_pady, padx=(10, 5))

entry_desired_z = ctk.CTkEntry(Desired_Pose_frame, width=desired_pose_entrt_width, font=desired_pose_entry_font)
entry_desired_z.grid(row=2, column=2, pady=desired_pose_pady, padx=(5, 10))

label_desired_theta = ctk.CTkLabel(Desired_Pose_frame, text="θᵣ: ", font=desired_pose_label_font)
label_desired_theta.grid(row=3, column=0, pady=desired_pose_pady, padx=(10, 5))

entry_desired_theta = ctk.CTkEntry(Desired_Pose_frame, width=desired_pose_entrt_width, font=desired_pose_entry_font)
entry_desired_theta.grid(row=4, column=0, pady=desired_pose_pady + 3, padx=(5, 10))

label_desired_phi = ctk.CTkLabel(Desired_Pose_frame, text="ϕᵣ: ", font=desired_pose_label_font)
label_desired_phi.grid(row=3, column=1, pady=desired_pose_pady, padx=(10, 5))

entry_desired_phi = ctk.CTkEntry(Desired_Pose_frame, width=desired_pose_entrt_width, font=desired_pose_entry_font)
entry_desired_phi.grid(row=4, column=1, pady=desired_pose_pady + 3, padx=(5, 10))

label_desired_psi = ctk.CTkLabel(Desired_Pose_frame, text="ψᵣ: ", font=desired_pose_label_font)
label_desired_psi.grid(row=3, column=2, pady=desired_pose_pady, padx=(10, 5))

entry_desired_psi = ctk.CTkEntry(Desired_Pose_frame, width=desired_pose_entrt_width, font=desired_pose_entry_font)
entry_desired_psi.grid(row=4, column=2, pady=desired_pose_pady + 3, padx=(5, 10))

send_button = ctk.CTkButton(Desired_Pose_frame, text="Send Pose", command=send_data_GUI)
send_button.grid(row=5, column=1, pady=20, padx=(5, 10))



stop_event = threading.Event()

#Control Frame
Control_frame = ctk.CTkFrame(root, width = 300, height = 300, corner_radius=10, fg_color='gray', border_color= 'green', border_width=2)
Control_frame.place(x=20,y=400)
label_Control_frame = ctk.CTkLabel(Control_frame, text="Control", font=('Times New Roman', 24))
label_Control_frame.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

stop_button = ctk.CTkButton(Control_frame, text="Stop", command=stop_trajectory)
stop_button.grid(row=1, column=0, pady=7, padx=(5, 10))

control_options = ["open loop","closed loop", "Data Driven", "Adaptive Data Driven"]
choose_control_combobox = ctk.CTkComboBox(Control_frame, width = 140, height = 28, corner_radius=10,border_width=5, values=control_options)
choose_control_combobox.grid(row=2, column=0, pady=7, padx=(5, 10))


#trajectory frame
Trajectory_frame = ctk.CTkFrame(root, width = 300, height = 300, corner_radius=10, fg_color='gray', border_color= 'green', border_width=2)
Trajectory_frame.place(x=190,y=400)
label_Trajectory_frame = ctk.CTkLabel(Trajectory_frame, text="Trajectory", font=('Times New Roman', 24))
label_Trajectory_frame.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

main_trajectory_buttom = ctk.CTkButton(Trajectory_frame, text="follow trajectory", command=start_trajectory_thread)
main_trajectory_buttom.grid(row=1, column=0, pady=7, padx=(5, 10))

options = ["main Trajectory", "inf Trajectory", "Flower Trajectory", "Archemides Spiral","Consecutive Triangles","consecutive merged steps"]
choose_trajectory_combobox = ctk.CTkComboBox(Trajectory_frame, width = 140, height = 28, corner_radius=10,border_width=5, values=options)
choose_trajectory_combobox.grid(row=2, column=0, pady=7, padx=(5, 10))


#save file text box (filname)
label_save = ctk.CTkLabel(root, text="Data Directory", font=('Times New Roman', 16))
label_save.place(x=505,y=500)
file_name_textBox = ctk.CTkTextbox(root, width = 140, height = 28, corner_radius=10,border_width=5)
file_name_textBox.place(x=500,y=530)
file_name_textBox.insert("1.0", "File Name")




#Backlash frame
Backlash_frame = ctk.CTkFrame(root, width = 300, height = 300, corner_radius=10, fg_color='gray', border_color= 'green', border_width=2)
Backlash_frame.place(x=355,y=400)
label_Backlash_frame = ctk.CTkLabel(Backlash_frame, text="Backlash", font=('Times New Roman', 24))
label_Backlash_frame.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

backlash_enable = tk.BooleanVar()
backlash_check_box = ctk.CTkCheckBox(Backlash_frame, width = 100, height=24, text="Apply Backlash", variable=backlash_enable)
backlash_check_box.grid(row=1, column=0, pady=10, padx=10, columnspan=3)

entry_backlash1 = ctk.CTkEntry(Backlash_frame, width=30, font=desired_pose_entry_font)
entry_backlash1.grid(row=2, column=0, pady=desired_pose_pady, padx=(5, 10))

entry_backlash2 = ctk.CTkEntry(Backlash_frame, width=30, font=desired_pose_entry_font)
entry_backlash2.grid(row=2, column=1, pady=desired_pose_pady, padx=(5, 10))

entry_backlash3 = ctk.CTkEntry(Backlash_frame, width=30, font=desired_pose_entry_font)
entry_backlash3.grid(row=2, column=2, pady=desired_pose_pady, padx=(5, 10))

entry_backlash4 = ctk.CTkEntry(Backlash_frame, width=30, font=desired_pose_entry_font)
entry_backlash4.grid(row=3, column=0, pady=desired_pose_pady, padx=(5, 10))

entry_backlash5 = ctk.CTkEntry(Backlash_frame, width=30, font=desired_pose_entry_font)
entry_backlash5.grid(row=3, column=1, pady=desired_pose_pady, padx=(5, 10))

entry_backlash6 = ctk.CTkEntry(Backlash_frame, width=30, font=desired_pose_entry_font)
entry_backlash6.grid(row=3, column=2, pady=desired_pose_pady, padx=(5, 10))







#Plotting Data Frame
def plot_trajectory_and_errors(file_name):
    df = pd.read_excel(file_name).iloc[5:]

    desired_data = df[['Desired_x', 'Desired_y', 'Desired_z', 'Desired_theta', 'Desired_phi', 'Desired_psi']]
    real_data = df[['EndEffector_x', 'EndEffector_y', 'EndEffector_z', 'EndEffector_theta', 'EndEffector_phi', 'EndEffector_psi']]

    #calculate errors
    errors = pd.DataFrame()
    for (desired_col, real_col) in zip(['Desired_x', 'Desired_y', 'Desired_z', 'Desired_theta', 'Desired_phi', 'Desired_psi'],
                                       ['EndEffector_x', 'EndEffector_y', 'EndEffector_z', 'EndEffector_theta', 'EndEffector_phi', 'EndEffector_psi']):
        errors[f'Error {desired_col.split("_")[1]}'] = desired_data[desired_col] - real_data[real_col]

    #convert index to numpy array for plotting
    time_index = desired_data.index.to_numpy()

    fig = plt.figure(figsize=(12, 24))

    #create subplots for desired vs real and error
    for i, (desired_col, real_col) in enumerate(zip(['Desired_x', 'Desired_y', 'Desired_z', 'Desired_theta', 'Desired_phi', 'Desired_psi'],
                                                    ['EndEffector_x', 'EndEffector_y', 'EndEffector_z', 'EndEffector_theta', 'EndEffector_phi', 'EndEffector_psi'])):
        ax1 = fig.add_subplot(6, 2, 2 * i + 1)  # 2*i + 1 for left column
        ax1.plot(time_index, desired_data[desired_col].to_numpy(), label=f'Desired {desired_col.split("_")[1]}')
        ax1.plot(time_index, real_data[real_col].to_numpy(), label=f'Real {real_col.split("_")[1]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{desired_col.split("_")[1]}')
        ax1.legend()

    # Plot errors
    for i, error_col in enumerate(errors.columns):
        # Right column for errors
        ax2 = fig.add_subplot(6, 2, 2 * i + 2)  # 2*i + 2 for right column
        ax2.plot(time_index, errors[error_col].to_numpy(), label=error_col)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error')
        ax2.legend()

    # Adjust layout
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    plot_file_name = f'{base_name}.png'
    plt.savefig(plot_file_name, dpi=300)
    plt.show()

    svg_file_name = f'{base_name}.svg'
    fig.savefig(svg_file_name)
def on_plot_button_click():
    file_name = filename_global
    threading.Thread(target=plot_trajectory_and_errors, args=(file_name,)).start()

DataPlotting_frame = ctk.CTkFrame(root, width = 100, height = 100, corner_radius=10, fg_color='gray', border_color= 'green', border_width=2)
plot_button = ctk.CTkButton(root, text="Plot Data", command=on_plot_button_click)
plot_button.place(x=660,y=533)








#Camera frame
Camera_frame = ctk.CTkFrame(root, width=640, height=480, corner_radius=10, fg_color="gray", border_color="red", border_width=2,)
Camera_frame.place(x=500, y=0)

canvas_video = tk.Canvas(Camera_frame, width=640, height=480)
canvas_video.pack()

calibrate_buttom = ctk.CTkButton(root, text="calibrate", command=calibrate)
calibrate_buttom.place(x=1000,y=530)






#pose frame: 
Pose_frame = ctk.CTkFrame(root, width=40, height=40, corner_radius=10, fg_color="gray", border_color="red", border_width=2)
Pose_frame.place(x=20, y=280)

Pose_label = ctk.CTkLabel(Pose_frame, text="Pose: ", font=('Times New Roman', 24))
Pose_label.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

pose_pady = 2
pose_padx = 2
pose_font = ('Times New Roman', 16)
pose_label_font = ('Times New Roman', 18)

label_xyz = ctk.CTkLabel(Pose_frame, text="x, y, z (cm): ", font=pose_label_font)
label_xyz.grid(row=1, column=0, pady=pose_pady, padx=pose_padx)

label_position = ctk.CTkLabel(Pose_frame, text="", font=pose_font)
label_position.grid(row=1, column=1, pady=pose_pady, padx=pose_padx)

label_angles_text = ctk.CTkLabel(Pose_frame, text="θ, ϕ, ψ (deg): ", font=pose_label_font)
label_angles_text.grid(row=2, column=0, pady=pose_pady, padx=pose_padx)

label_angles = ctk.CTkLabel(Pose_frame, text="", font=pose_font)
label_angles.grid(row=2, column=1, pady=pose_pady, padx=pose_padx)





def update_labels():
    global end_effector_POSE_base_global

    #print (end_effector_POSE_base_global)
    # Update the position label
    position_text = f"x: {end_effector_POSE_base_global[0]:.2f}, y: {end_effector_POSE_base_global[1]:.2f}, z: {end_effector_POSE_base_global[2]:.2f}"
    label_position.configure(text=position_text)
    
    # Update the angles label
    angles_text = f"θ: {end_effector_POSE_base_global[3]:.2f}, ϕ: {end_effector_POSE_base_global[4]:.2f}, ψ: {end_effector_POSE_base_global[5]:.2f}"
    label_angles.configure(text=angles_text)
    
    # Call this function again after 1000 milliseconds (1 second)
    root.after(250, update_labels)

update_labels()

def update_image(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        canvas_video.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas_video.imgtk = imgtk

def update_frame():
    frame = camera.get_frame()
    if frame is not None:
        frame = estimate_pose(frame)
        update_image(frame)
        
    canvas_video.after(3, update_frame)

camera = CameraCapture()
camera.start()
update_frame()
# Run the application
root.mainloop()
camera.stop()
current_measurements_global.stop()


