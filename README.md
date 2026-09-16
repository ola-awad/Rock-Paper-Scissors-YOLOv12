# ✂️ Rock Paper Scissors Detection using YOLOv12

A real-time hand gesture recognition system designed to detect and classify **Rock**, **Paper**, and **Scissors** gestures using the latest **YOLOv12** computer vision model and OpenCV.

---

## 📌 Features
- **Real-Time Detection:** Smooth webcam inference with bounded Region of Interest (ROI) filtering.
- **Custom Trained Model:** Fine-tuned YOLOv12 architecture for high-accuracy gesture classification.
- **Easy Deployment:** Python inference script ready for immediate testing.

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| `best.pt` | Trained YOLOv12 model weights (~5.3 MB) |
| `Rock_Paper_Scissor.ipynb` | Jupyter Notebook containing dataset processing and model training |
| `Rock.py` | Python script for real-time webcam inference |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Rock-Paper-Scissors-YOLOv12.git](https://github.com/YOUR_USERNAME/Rock-Paper-Scissors-YOLOv12.git)
   cd Rock-Paper-Scissors-YOLOv12

2-**Install required dependencies:**
   pip install ultralytics opencv-python

  ** 🚀 How to Run**
   To test real-time detection via webcam, run:
     python Rock.py
