import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder


BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from thresholds import get_thresholds_beginner_side, get_thresholds_pro_side, \
                       get_thresholds_beginner_front, get_thresholds_pro_front # Import new threshold functions
from process_frame_side import ProcessFrameSide 
from process_frame_front import ProcessFrameFront

st.title('AI Fitness Trainer: Squats Analysis')

mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)
view = st.radio('Select View', ['Side View'], horizontal=True) 

# view = st.radio('Select View', ['Side View', 'Front View'], horizontal=True) # front view is still under development

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


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="rgb24")  # Decode and get RGB frame
    frame, _ = processor.process(frame, pose)  # Process frame
    return av.VideoFrame.from_ndarray(frame, format="rgb24")  # Encode and return BGR frame

def out_recorder_factory() -> MediaRecorder:
    return MediaRecorder(output_video_file)


ctx = webrtc_streamer(
                        key="Squats-pose-analysis",
                        video_frame_callback=video_frame_callback,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        media_stream_constraints={"video": {"width": {'min': 480, 'ideal': 480}}, "audio": False}, # Audio is False here, meaning microphone audio is not captured.
                        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=False),
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