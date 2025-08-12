# 🏋️‍♂️ AI Fitness Trainer – Squat Analysis

## 📌 Overview
The **AI Fitness Trainer – Squat Analysis** is a computer vision-based application that evaluates squat form in real-time using **pose estimation**.  
Currently, **side view squat analysis is fully functional**, while **front view analysis is under development**.  
The system detects body keypoints, analyses posture, and provides instant feedback to help improve technique and prevent injuries.

---

## 🚀 Features
- ✅ **Real-time Pose Estimation** using OpenCV & MediaPipe  
- ✅ **Side View Analysis** – accurate squat depth and posture detection  
- ⚙️ **Front View Analysis** – *in progress*  
- **Form Feedback** – detects incorrect posture and depth  
- **Repetition Counter** – counts completed squats  
- Lightweight and runs on CPU

---

## 🔧 Tech Stack
- **Python**
- **OpenCV**
- **MediaPipe Pose**
- **NumPy**
- **Matplotlib** (for visualization)
- **Streamlit** (optional UI integration)

---

## 📂 Installation
```bash
# Clone the repository
git clone https://github.com/azmeera-qureshi/ai-fitness-squat-analysis.git
cd ai-fitness-squat-analysis

# Install dependencies
pip install -r requirements.txt


```

### Execution
In terminal run following command

```
streamlit run 🏠️_Demo.py
```
