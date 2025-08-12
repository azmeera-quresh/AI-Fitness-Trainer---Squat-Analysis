import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line

class ProcessFrameFront:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }

        # Landmark features
        self.dict_features = {
            'left': {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27, 'foot': 31},
            'right': {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28, 'foot': 32},
            'nose': 0
        }

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {

            'state_seq': [], 
            'start_inactive_time': time.perf_counter(), 
            'start_inactive_time_front': time.perf_counter(), # without that getting blank
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0, # without that getting blank
            'DISPLAY_TEXT' : np.full((5,), False), 
            'COUNT_FRAMES' : np.zeros((5,), dtype=np.int64),
            'INCORRECT_POSTURE': False, 
            'prev_state': None, 
            'curr_state':None, 
            'SQUAT_COUNT': 0, 
            'IMPROPER_SQUAT':0,
            'last_valid_pose_time': time.perf_counter() 
        }
        
        self.FEEDBACK_ID_MAP = {
            0: ('KNEES CAVING IN', 215, (255, 80, 80)),
            1: ('KNEES TOO WIDE', 170, (255, 80, 80)),
            2: ('SQUAT TOO DEEP', 125, (255, 80, 80)),
            3: ('SQUAT TOO SHALLOW', 80, (255, 80, 80)),
            4: ('Resetting COUNTERS due to inactivity!!!', 135, (255, 100, 100))
            }
    def _get_state(self, hip_knee_dist_normalized):
        """Determine squat state based on hip-knee distance."""
        if hip_knee_dist_normalized > self.thresholds['SQUAT_SHALLOW_THRESH']:
            return 's1'  # Standing
        elif self.thresholds['SQUAT_DEEP_THRESH'] < hip_knee_dist_normalized <= self.thresholds['SQUAT_SHALLOW_THRESH']:
            return 's2'  # Mid-squat
        else:
            return 's3'  # Deep squat

    def _update_state_sequence(self, state):
        """Update state sequence for rep counting."""
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 0)) or \
               (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 1)):
                self.state_tracker['state_seq'].append(state)
        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        for idx in np.where(c_frame)[0]:
            if idx in dict_maps:
                draw_text(
                        frame, 
                        dict_maps[idx][0], 
                        pos=(30, dict_maps[idx][1]),
                        text_color=(255, 255, 230),
                        font_scale=0.6,
                        text_color_bg=dict_maps[idx][2]
                    )
        return frame

    def _draw_skeleton(self, frame, landmarks, color=(0, 255, 255), thickness=4):
        # Draw all connections for a complete skeleton
        connections = [
            # Face (not really needed for squats)
            # Body
            (11, 12),  # Shoulders
            (12, 24),  # Right shoulder to right hip
            (11, 23),  # Left shoulder to left hip
            (23, 24),  # Hips
            
            # Left arm
            (11, 13),  # Left shoulder to left elbow
            (13, 15),  # Left elbow to left wrist
            
            # Right arm
            (12, 14),  # Right shoulder to right elbow
            (14, 16),  # Right elbow to right wrist
            
            # Left leg
            (23, 25),  # Left hip to left knee
            (25, 27),  # Left knee to left ankle
            (27, 31),  # Left ankle to left foot
            
            # Right leg
            (24, 26),  # Right hip to right knee
            (26, 28),  # Right knee to right ankle
            (28, 32)   # Right ankle to right foot
        ]
        
        # frame_height, frame_width = frame.shape[:2]
        
        for (start_idx, end_idx) in connections:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            if start.visibility > 0.1 and end.visibility > 0.1:
                start_coord = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_coord = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_coord, end_coord, self.COLORS['light_blue'], 4, self.linetype)
                cv2.circle(frame, start_coord, 7, self.COLORS['yellow'], -1, self.linetype)
                cv2.circle(frame, end_coord, 7, self.COLORS['yellow'], -1, self.linetype)
    
    def _is_skeleton_complete(self, landmarks, frame_width, frame_height):
        """Check if we have all the key points needed for analysis"""
        required_points = [
            self.dict_features['left']['shoulder'],
            self.dict_features['right']['shoulder'],
            self.dict_features['left']['hip'],
            self.dict_features['right']['hip'],
            self.dict_features['left']['knee'],
            self.dict_features['right']['knee'],
            self.dict_features['left']['ankle'],
            self.dict_features['right']['ankle']
        ]
        
        for point in required_points:
            landmark = landmarks.landmark[point]
            if landmark.visibility < 0.1:
                return False
                
            # Convert to pixel coordinates and check if they're within frame bounds
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            if not (0 <= x < frame_width and 0 <= y < frame_height):
                return False
                
        return True
    
    

    def process(self, frame: np.array, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape
        keypoints = pose.process(frame) # Process the image.
        # current_state = None
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            # Draw complete skeleton first
            self._draw_skeleton(frame, ps_lm, self.COLORS['light_blue'])

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            # Check if skeleton is complete (all key points visible)
            skeleton_complete = self._is_skeleton_complete(ps_lm, frame_width, frame_height)

            # Camera alignment check - only show message if skeleton is incomplete
            if not skeleton_complete:
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True
                    self.state_tracker['DISPLAY_TEXT'][4] = True # Show reset counters message

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                    

                # Update COUNT_FRAMES and display feedback if any DISPLAY_TEXT is True
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, False)

                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                
                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  
                
                draw_text(
                    frame, 
                    'BODY NOT FULLY VISIBLE!!!', 
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 
                
                draw_text(
                    frame, 
                    'OFFSET ANGLE: '+str(int(offset_angle)), # Cast to int for cleaner display
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 

                # Reset other state variables
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None
                
                # Update feedback displays
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, False)

                # Clear messages after threshold
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
            
            # Camera is aligned properly.
            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                

                # Reset incorrect posture flag for current frame before re-evaluation
                self.state_tracker['INCORRECT_POSTURE'] = False
                self.state_tracker['DISPLAY_TEXT'][:4] = False # Clear previous feedback messages (excluding reset)

                # 1. Knee Valgus Check
                # Calculate horizontal distance between knee and ankle for each leg
                left_knee_to_ankle_horiz_dist = abs(left_knee_coord[0] - left_ankle_coord[0])
                right_knee_to_ankle_horiz_dist = abs(right_knee_coord[0] - right_ankle_coord[0])

                # Use hip width as a reference for normalization
                hip_width = abs(left_hip_coord[0] - right_hip_coord[0])

                if hip_width > 10: # Ensure a meaningful hip width is detected
                    left_knee_valgus_ratio = left_knee_to_ankle_horiz_dist / hip_width
                    right_knee_valgus_ratio = right_knee_to_ankle_horiz_dist / hip_width

                    if left_knee_valgus_ratio < self.thresholds['KNEE_VALGUS_THRESH_PERCENT'] or \
                       right_knee_valgus_ratio < self.thresholds['KNEE_VALGUS_THRESH_PERCENT']:
                        self.state_tracker['DISPLAY_TEXT'][0] = True # Knees Caving In
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        play_sound = 'incorrect' # Play sound when error detected

                # 2. Hip Shift Check
                mid_shldr_x = (left_shldr_coord[0] + right_shldr_coord[0]) // 2
                mid_hip_x = (left_hip_coord[0] + right_hip_coord[0]) // 2
                
                # Check for significant horizontal shift between shoulder and hip centerlines
                if frame_width > 0:
                    hip_shift_percentage = abs(mid_shldr_x - mid_hip_x) / frame_width
                    if hip_shift_percentage > self.thresholds['HIP_SHIFT_THRESH_PERCENT']:
                        self.state_tracker['DISPLAY_TEXT'][1] = True # Shifting to One Side
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        play_sound = 'incorrect' # Play sound when error detected

                # 3. Shoulder-Hip Misalignment (Side bending/leaning)
                if abs(left_shldr_coord[0] - left_hip_coord[0]) > (hip_width * self.thresholds['SHOULDER_HIP_ALIGN_THRESH']) or \
                   abs(right_shldr_coord[0] - right_hip_coord[0]) > (hip_width * self.thresholds['SHOULDER_HIP_ALIGN_THRESH']):
                    self.state_tracker['DISPLAY_TEXT'][2] = True # Shoulder-Hip Misalignment
                    self.state_tracker['INCORRECT_POSTURE'] = True
                    play_sound = 'incorrect' # Play sound when error detected

                # --- Inactivity Check (General, not front-specific in this block) ---
                display_inactivity = False
                
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                self.state_tracker['start_inactive_time'] = end_time

                if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True
                    self.state_tracker['DISPLAY_TEXT'][4] = True # Show reset counters message

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                # ----------------------------------- Feedback and Display ---------------------------------------------
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, False)

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # Display counters
                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                
                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )  
                
                # Clear feedback messages after their display threshold
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                
                if self.state_tracker['INCORRECT_POSTURE'] and not self.state_tracker['prev_state'] == 'INCORRECT':
                    self.state_tracker['IMPROPER_SQUAT'] += 1
                    self.state_tracker['prev_state'] = 'INCORRECT'
                elif not self.state_tracker['INCORRECT_POSTURE']:
                    self.state_tracker['prev_state'] = 'CORRECT'

        else: # No pose landmarks detected
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                display_inactivity = True
                self.state_tracker['DISPLAY_TEXT'][4] = True #show reset counters

            self.state_tracker['start_inactive_time'] = end_time

            # Call _show_feedback here for general inactivity (no detections)
            self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1
            frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, False)

            draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                
            draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
            
            # Reset all other state variables for front view when no pose is detected
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['prev_state'] =  None

            # Clear all other feedback messages except for the reset message itself
            self.state_tracker['DISPLAY_TEXT'][:4] = False
            self.state_tracker['COUNT_FRAMES'][:4] = 0

            # Update DISPLAY_TEXT based on COUNT_FRAMES (specifically for index 4 here)
            self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
            self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
            
        return frame, play_sound