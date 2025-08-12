import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import base64

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from thresholds import get_thresholds_beginner_side, get_thresholds_pro_side, \
                       get_thresholds_beginner_front, get_thresholds_pro_front # Import new threshold functions
from process_frame_side import ProcessFrameSide 
from process_frame_front import ProcessFrameFront

st.title('AI Fitness Trainer: Squats Analysis')

mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)
view = st.radio('Select View', ['Side View', 'Front View'], horizontal=True) 

thresholds = None 

# Select thresholds based on both mode and view
if mode == 'Beginner':
    if view == 'Side View':
        thresholds = get_thresholds_beginner_side()
    else: # Front View
        thresholds = get_thresholds_beginner_front()
elif mode == 'Pro':
    if view == 'Side View':
        thresholds = get_thresholds_pro_side()
    else: # Front View
        thresholds = get_thresholds_pro_front()

# Initialize the processor based on the selected view
if view == 'Side View':
    processor = ProcessFrameSide(thresholds=thresholds, flip_frame=True)
else: # Front View
    processor = ProcessFrameFront(thresholds=thresholds, flip_frame=True)
    
pose = get_mediapipe_pose()

if 'download' not in st.session_state:
    st.session_state['download'] = False

output_video_file = f'output_live.flv'

# Define base64 encoded audio for different feedback sounds
AUDIO_FILES = {
    'bend_backwards': base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', 'bend_backwards.wav'), 'rb').read()).decode(),
    'bend_forward': base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', 'bend_forward.wav'), 'rb').read()).decode(),
    'knee_over_toe': base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', 'knee_over_toe.wav'), 'rb').read()).decode(),
    'squat_deep': base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', 'squat_deep.wav'), 'rb').read()).decode(),
    'incorrect_squat': base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', 'incorrect.wav'), 'rb').read()).decode(),
    'reset_counters': base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', 'reset_counters.wav'), 'rb').read()).decode(),
}

def play_sound(sound_key):
    if sound_key and sound_key in AUDIO_FILES:
        audio_b64 = AUDIO_FILES[sound_key]
        st.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True,
        )
    elif sound_key and sound_key.isdigit(): # Play sound for count
        audio_b64 = base64.b64encode(open(os.path.join(BASE_DIR, 'sounds', f'{sound_key}.wav'), 'rb').read()).decode()
        st.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True,
        )
if st.button("Test Sound"):
    play_sound("bend_forward")

def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="rgb24")
    img, sound = processor.process(img, pose)
    if sound:
        play_sound(sound)
    return av.VideoFrame.from_ndarray(img, format="rgb24")

def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(output_video_file)

ctx = webrtc_streamer(
                        key="Squats-pose-analysis",
                        video_frame_callback=video_frame_callback,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        media_stream_constraints={"video": {"width": {'min':480, 'ideal':480}}, "audio": False},
                        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
                        out_recorder_factory=out_recorder_factory
                    )

download_button = st.empty()

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data = op_vid, file_name='output_live.flv')

        if download:
            st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()

