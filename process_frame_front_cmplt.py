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
                print(f"Drawing connection from {start_idx} to {end_idx}")  
                print(f"Start: {start_coord}, End: {end_coord}")  
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
    
    def _validate_coords(self, *coords, frame_width, frame_height):
        """Helper to validate coordinates"""
        for i, coord in enumerate(coords):
            if not (0 <= coord[0] < frame_width and 0 <= coord[1] < frame_height):
                print(f"Invalid coord {i}: {coord} (frame: {frame_width}x{frame_height})")
                return False
        return True
    

    def process(self, frame: np.array, pose):
        display_inactivity = False
        play_sound = None
        frame_height, frame_width, _ = frame.shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB first
        output_frame = frame.copy()
        # Debug: Print frame dimensions
        print(f"Frame dimensions: {frame_width}x{frame_height}")
        
        keypoints = pose.process(frame_rgb) # Process the image.
        # current_state = None
        if keypoints.pose_landmarks:
            print("✅ Pose landmarks detected!")
            ps_lm = keypoints.pose_landmarks
            # Debug: Print first few landmark coordinates
            for i, landmark in enumerate(ps_lm.landmark[:5]):  # Just first 5 for brevity
                print(f"Landmark {i}: X={landmark.x:.2f}, Y={landmark.y:.2f}, Visibility={landmark.visibility:.2f}")
            # Draw complete skeleton first
            self._draw_skeleton(output_frame, ps_lm, self.COLORS['light_blue'])
            self.state_tracker['last_valid_pose_time'] = time.perf_counter()

            # Get landmarks
            # left_hip = get_landmark_features(ps_lm.landmark, self.dict_features, 'left', left_hip_coord, frame_width, frame_height)
            # right_hip = get_landmark_features(ps_lm.landmark, self.dict_features, 'right', right_hip_coord, frame_width, frame_height)
            # left_knee = get_landmark_features(ps_lm.landmark, self.dict_features, 'left', left_knee_coord, frame_width, frame_height)
            # right_knee = get_landmark_features(ps_lm.landmark, self.dict_features, 'right', right_knee_coord, frame_width, frame_height)
            
            # --- Camera Alignment Check ---
            # nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            # left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
            #                     get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            # right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
            #                     get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)
            
                
            # offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)
            
            # Get landmarks
            try:
                left_hip = get_landmark_features(ps_lm.landmark, self.dict_features, 'left', 'hip', frame_width, frame_height)
                right_hip = get_landmark_features(ps_lm.landmark, self.dict_features, 'right', 'hip', frame_width, frame_height)
                left_knee = get_landmark_features(ps_lm.landmark, self.dict_features, 'left', 'knee', frame_width, frame_height)
                right_knee = get_landmark_features(ps_lm.landmark, self.dict_features, 'right', 'knee', frame_width, frame_height)
                
                # Camera Alignment Check
                nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
                left_shldr_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'left', 'shoulder', frame_width, frame_height)
                right_shldr_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'right', 'shoulder', frame_width, frame_height)
                
                # Debug prints
                print(f"Left hip: {left_hip}, Right hip: {right_hip}")
                print(f"Left knee: {left_knee}, Right knee: {right_knee}")
                print(f"Shoulders - Left: {left_shldr_coord}, Right: {right_shldr_coord}")
                if not self._validate_coords(left_hip, right_hip, left_knee, right_knee,
                                      nose_coord, left_shldr_coord, right_shldr_coord,
                                      frame_width=frame_width, frame_height=frame_height):
                    print("Invalid coordinates - skipping frame")
                    return frame, None
                
            except (ValueError, KeyError) as e:
                print(f"Landmark error: {str(e)}")
                return frame, None
            
        
            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            # Camera alignment check
            if offset_angle > self.thresholds['OFFSET_THRESH'] or not self._is_skeleton_complete(ps_lm, frame_width, frame_height):
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

                # if self.flip_frame:
                #     frame = cv2.flip(frame, 1)

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
                    'CAMERA NOT ALIGNED PROPERLY OR BODY NOT FULLY VISIBLE!!!', 
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
                
                return frame, play_sound
        
            # Camera is aligned properly.
            else:
                print(f"Input frame - Type: {frame.dtype}, Shape: {frame.shape}, Max: {frame.max()}, Min: {frame.min()}")
                 # --- Squat Analysis (ONLY if camera is aligned) ---
                hip_knee_dist = (abs(left_hip[1] - left_knee[1]) + abs(right_hip[1] - right_knee[1])) / 2
                hip_knee_dist_normalized = hip_knee_dist / frame_height
                current_state = self._get_state(hip_knee_dist_normalized)
                
                # Update state sequence
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)
                
                # --- Error Detection ---
                self.state_tracker['DISPLAY_TEXT'][:4] = False  # Reset previous errors
                hip_width = abs(left_hip[0] - right_hip[0])
                knee_distance = abs(left_knee[0] - right_knee[0])
                knee_ratio = knee_distance / hip_width

                if knee_ratio < self.thresholds['KNEE_VALGUS_THRESH_PERCENT']:
                    self.state_tracker['DISPLAY_TEXT'][0] = True  # Knees caving
                elif knee_ratio > self.thresholds['KNEE_DISTANCE_MAX_PERCENT']:
                    self.state_tracker['DISPLAY_TEXT'][1] = True  # Knees too wide

                if hip_knee_dist_normalized < self.thresholds['SQUAT_DEEP_THRESH']:
                    self.state_tracker['DISPLAY_TEXT'][2] = True  # Too deep
                elif hip_knee_dist_normalized > self.thresholds['SQUAT_SHALLOW_THRESH']:
                    self.state_tracker['DISPLAY_TEXT'][3] = True  # Too shallow

                # --- Counter Logic ---
                if current_state == 's1':
                    if len(self.state_tracker['state_seq']) == 3 and not any(self.state_tracker['DISPLAY_TEXT'][:4]):
                        self.state_tracker['SQUAT_COUNT'] += 1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])
                    elif any(self.state_tracker['DISPLAY_TEXT'][:4]):
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'incorrect'
                    self.state_tracker['state_seq'] = []
                

                # ----------------------------------- Feedback and Display ---------------------------------------------
                # self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1
                # frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, False)

                # if display_inactivity:
                #     play_sound = 'reset_counters'
                #     self.state_tracker['start_inactive_time'] = time.perf_counter()
                #     self.state_tracker['INACTIVE_TIME'] = 0.0

                # # Display counters
                # draw_text(
                #     frame, 
                #     "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                #     pos=(int(frame_width*0.68), 30),
                #     text_color=(255, 255, 230),
                #     font_scale=0.7,
                #     text_color_bg=(18, 185, 0)
                # )  
                
                # draw_text(
                #     frame, 
                #     "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                #     pos=(int(frame_width*0.68), 80),
                #     text_color=(255, 255, 230),
                #     font_scale=0.7,
                #     text_color_bg=(221, 0, 0),
                # )  
                
                # # Clear feedback messages after their display threshold
                # self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                # self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                
                # if self.state_tracker['INCORRECT_POSTURE'] and not self.state_tracker['prev_state'] == 'INCORRECT':
                #     self.state_tracker['IMPROPER_SQUAT'] += 1
                #     self.state_tracker['prev_state'] = 'INCORRECT'
                # elif not self.state_tracker['INCORRECT_POSTURE']:
                #     self.state_tracker['prev_state'] = 'CORRECT'

        else: # No pose landmarks detected
            print("❌ No pose landmarks detected in this frame!")
            inactive_time = time.perf_counter() - self.state_tracker['last_valid_pose_time']
            if inactive_time >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                self.state_tracker['DISPLAY_TEXT'][4] = True
                play_sound = 'reset_counters'

        # --- Display Feedback ---
        self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
        frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, False)

        # Clear messages after threshold
        mask = self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']
        self.state_tracker['DISPLAY_TEXT'][mask] = False
        self.state_tracker['COUNT_FRAMES'][mask] = 0

        # Display counters
        draw_text(frame, f"CORRECT: {self.state_tracker['SQUAT_COUNT']}", 
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0))
        
        draw_text(frame, f"INCORRECT: {self.state_tracker['IMPROPER_SQUAT']}", 
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0))

        if self.flip_frame:
            output_frame  = cv2.flip(output_frame , 1)

        return output_frame, play_sound