def get_thresholds_beginner_side():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (70, 95)
                           }    

    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
                    'HIP_THRESH'   : [10, 50],
                    'ANKLE_THRESH' : 45,
                    'KNEE_THRESH'  : [50, 70, 95],
                    'OFFSET_THRESH'    : 35.0, 
                    'INACTIVE_THRESH'  : 15.0,
                    'CNT_FRAME_THRESH' : 50
                }

    return thresholds

def get_thresholds_pro_side():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (80, 95)
                           }    

    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
                    'HIP_THRESH'   : [15, 50],
                    'ANKLE_THRESH' : 30,
                    'KNEE_THRESH'  : [50, 80, 95],
                    'OFFSET_THRESH'    : 35.0, 
                    'INACTIVE_THRESH'  : 15.0,
                    'CNT_FRAME_THRESH' : 50
                 }
                 
    return thresholds


def get_thresholds_beginner_front():
    thresholds = {
        'KNEE_VALGUS_THRESH_PERCENT': 0.15,  # Min knee distance (% of hip width)
        'KNEE_DISTANCE_MAX_PERCENT': 0.30,   # Max knee distance (% of hip width)
        'SQUAT_DEEP_THRESH': 0.15,           # Min hip-knee distance (% of frame height)
        'SQUAT_SHALLOW_THRESH': 0.30,         # Max hip-knee distance (% of frame height)
        'OFFSET_THRESH': 35.0, 
        'INACTIVE_THRESH': 15.0, 
        'CNT_FRAME_THRESH': 50, 
        # 'KNEE_VALGUS_THRESH_PERCENT': 0.15, # Max allowable horizontal distance between knee and ankle as % of hip width
        }
    return thresholds


def get_thresholds_pro_front():
    thresholds = {
        'OFFSET_THRESH': 35.0, 
        'INACTIVE_THRESH': 15.0, 
        'CNT_FRAME_THRESH': 50, #frame count
        'KNEE_VALGUS_THRESH_PERCENT': 0.10, # Tighter threshold for pro mode
        'KNEE_DISTANCE_MAX_PERCENT': 0.30,   # Max knee distance (% of hip width)
        'SQUAT_DEEP_THRESH': 0.15,           # Min hip-knee distance (% of frame height)
        'SQUAT_SHALLOW_THRESH': 0.30,         # Max hip-knee distance (% of frame height)
        
    }
    return thresholds

