# virtual-digital-human-project
virtual digital human project using Mediapipe and Unitychan
# Workflow
## 1. **Real-time capture of facial landmarks  based on MediaPipe**
We used MediaPipe and OpenCV to
detect 468 facial landmarks on the user's face in real-time. The facial
landmarks are points on the front that can be used to define the shape and structure of
the face, such as the corners of the mouth, the tip of the nose, and the edges of the
eyebrows. These landmarks include points on the eyes, eyebrows, nose, mouth, and
other facial features. We also computed the mouth aspect ratio (MAR) and eye aspect
ratio (EAR) from these landmarks.
## 2. **Real-time Head Pose Estimation based on the facial landmarks**
We used the function solvePnP provided by
OpenCV to compute the rotation vector based on the 468 facial. Once we had the rotation vector, we used it to animate the virtual human model in
Unity. Specifically, we used the pitch, yaw, and roll values to rotate the head of the
virtual human model in the corresponding directions. This allowed us to achieve
smooth and realistic head movements in the virtual human model that were
synchronized with the activities of the user's head in the real world.
landmarks.
## 3. **Facial expression recognition based on the facial landmarks**
To recognize the user's facial expression, we
labeled the 468 facial landmarks with specific facial expressions ('Angry', 'Happy',
'Neutral', 'Sad', 'Surprise') and created a dataset in .csv format. We then used
TensorFlow.keras to train a machine learning model on this data. The model is a
sequential neural network with 8 dense layers and a dropout layer. It achieved an
accuracy of 98.84% after 500 epochs.
## 4. **Real-time data transmission to virtual human model based on TCP**
To transmit the data
collected from the real-time capture and facial expression recognition to the virtual
human model in Unity, we used TCP (Transmission Control Protocol). To establish a TCP connection, we first initialized a TcpListener on the Unity side, which listens for incoming connections on a specific IP address and port number. On
the Python side, we established a TcpClient and connected it to the TcpListener using
the same IP address and port number. To send data from Python to Unity, we used the send method of the TcpClient and
passed it a byte array containing the data to be transmitted. On the Unity side, we used
the NetworkStream's Read method to read the incoming data and convert it into a byte
array. We then parsed the byte array to extract the relevant data and used it to drive
the virtual human model. 
## 5. **Drive the virtual human model based on the transmission data**
We used the Animator component and Skinned Mesh Renderer component to control
the animation of the virtual human model based on the data transmitted from the
real-time capture and facial expression recognition steps. To rotate the head, we used the rotation vector to compute the pitch, yaw, and roll
angles of the head, and applied these angles as rotations to the neck bone of the virtual
human model. To adjust the facial features, we used the MAR and EAR values to set
the blend shapes (i.e., morph targets) of the mouth and eyes, respectively. This
allowed us to animate the virtual human model's facial features in real-time based on
the user's facial expressions.
## 6. **Model optimization**
To further optimize the virtual human model, we implemented two additional
techniques:the **Kalman filter** and **the amount of rotation phasor change detection**
### 6.1 Using a Kalman filter to stabilize the facial landmarks 
The Kalman filter is a mathematical method that combines multiple sources of data (e.g., the facial
landmarks and the previous state of the head pose) to estimate the true state of a
system (e.g., the head pose of the user) with a higher degree of accuracy. It works
by using a state transition model to predict the next state of the system based on
the current state and an input control vector, and a measurement model to update
the prediction based on the observed data. This allows the Kalman filter to reduce
the noise and uncertainty in the data and produce a more stable estimate of the
true state of the system.
### 6.2 The amount of rotation phasor change detection 
This technique measures the amount of change in the rotation of the head from one frame to the next
and uses this information to stabilize the virtual human model. It works by calculating
the phasor change of the head pose in each frame, which is a complex number that
represents the angle and magnitude of the rotation. If the phasor change is above a
certain threshold, it means that the head has undergone a significant rotation and the
virtual human model should be adjusted accordingly. If the phasor change is below
the threshold, it means that the head has not undergone a significant rotation and the
virtual human model can be left unchanged. This helps to reduce the jitter and
instability in the animation and produce a more realistic and smooth motion.
 
# Demo
![Demo](https://raw.githubusercontent.com/shenllyz/virtual-digital-human-project/main/demo.gif)

# References/ Credits

Detect 468 Face Landmarks in Real-time | OpenCV Python | 
|Project|Author|LICENSE|
|:-|:-:|-:|
|[VTuber-Python-Unity](https://github.com/mmmmmm44/VTuber-Python-Unity)|[mmmmmm44](https://github.com/mmmmmm44)|[LICENSE](https://github.com/mmmmmm44/VTuber-Python-Unity/blob/main/LICENSE)|
|[VTuber_Unity](https://github.com/kwea123/VTuber_Unity)|[AI葵](https://github.com/kwea123)|[LICENSE](https://github.com/kwea123/VTuber_Unity/blob/master/LICENSE)|
|[Facial-emotion-recognition-using-mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe)|[ REWTAO ](https://github.com/REWTAO)|[LICENSE](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe/blob/main/LICENSE)|
 
 
 
