# HandGesturesRecognition-For-TritonDroids-Humanoid-
I am experimenting with Hand Gesture Recorgnition so that the Humanoid we are building can have features where it recognizes and responds to hand signals. (This is much in the future but might as well experiment:) )

## 🎯 How It Works  
The **hand gesture recognition system** consists of several key stages:  

### 1️⃣ Background Calibration  
📌 The system **captures a static background** during the first few frames to later detect hand movements by comparing new frames against this baseline.  

- Uses **Gaussian Blur** for noise reduction.  
- Computes a **weighted average** of the background to enhance stability.  

### 2️⃣ Region of Interest (ROI) Extraction  
📌 Instead of processing the entire frame, the system focuses on a **specific region** where the hand is expected to be.  

- The **top-right portion** of the frame is selected.  
- Helps **reduce computation time** and **improve accuracy**.  

### 3️⃣ Hand Segmentation  
📌 The system isolates the hand from the background using **image differencing and thresholding**.  

- Computes the **absolute difference** between the current frame and the stored background.  
- Applies **binary thresholding** to remove noise and extract the hand silhouette.  
- Identifies the **largest contour**, assuming it's the hand.  

### 4️⃣ Convex Hull & Contour Analysis  
📌 After segmentation, the system determines hand shape using:  

- **Convex Hull Detection:** Identifies the outer boundary of the hand.  
- **Contour Analysis:** Extracts **hand extremities** (top, bottom, left, right).  
- **Center Detection:** Finds the center of the palm for motion tracking.  

### 5️⃣ Gesture Classification  
📌 Based on the number of extended fingers and motion detection, the system classifies hand gestures:  

| Gesture       | Description |
|--------------|-------------|
| ✊ Rock       | No fingers extended |
| ☝ Pointing   | One finger extended |
| ✌ Scissors   | Two fingers extended |
| 👋 Waving    | Motion detected across frames |

#### **Finger Counting Algorithm:**  
- Draws a **horizontal reference line** across the fingers.  
- Counts **contour intersections** with this line.  

#### **Motion Tracking for Waving:**  
- Measures **frame-to-frame movement** of the hand’s center position.  
- If movement exceeds a threshold, it is classified as **waving**.  

### 6️⃣ Display & Output  
📌 The recognized gesture is displayed on-screen with:  

- **Text overlay** indicating the gesture.  
- **Highlighted ROI** showing the detection area.  
- Press **'x'** to exit the program.  

---

## 🎮 Usage  
1. **Start the program** – The system will calibrate the background for the first few seconds.  
2. **Place your hand inside the ROI** (highlighted area) for detection.  
3. **Move your hand or form gestures** – The system will classify them in real-time.  
4. **Exit the program** by pressing **'x'**.  

---

## 🏗️ Technologies Used  
- **Python** (OpenCV, NumPy)  
- **Computer Vision** (Edge Detection, Contour Analysis, Thresholding)  
- **Machine Perception** (Convex Hull, Hand Segmentation, Background Subtraction)  

---

## 📈 Future Improvements  
🚀 **Deep Learning Integration** - Train CNNs for more advanced gesture recognition  
🚀 **Multi-Hand Tracking** - Support for multiple hands simultaneously  
🚀 **Expanded Gesture Set** - Detect more complex hand movements  
