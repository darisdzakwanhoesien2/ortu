import cv2
import mediapipe as mp
import specific_landmarks as landmarks
import numpy as np
import time
import csv
import threading
from playsound import playsound

font = cv2.FONT_HERSHEY_SIMPLEX

def play_sound(sound_path):
    playsound(sound_path)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
vidcap = cv2.VideoCapture(0)

specific_face_landmark_id = landmarks.user_specific()
specific_hand_landmark_id = 8  # The mediapipe
sound_path = "statics/Instrumental Music [ No Copyright ] Sholawat Nariyah  Track 01.mp3"

# Initialize variables for time tracking
start_time = time.time()  # Initialize start_time here

# Set the durations for video processing (in seconds) as a list
processing_durations = [10, 15]
processing = 0

# Run the video processing loop for each duration
for processing_duration in processing_durations:
    accurate_lap_times = [0]
    current_max = 0

    # Run the video processing loop
    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
        while True:
            ret, frame = vidcap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            # + face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            for face_landmark_id in specific_face_landmark_id:
                if results.face_landmarks:
                    specific_face_landmark = results.face_landmarks.landmark[face_landmark_id]
                    h, w, c = image.shape
                    cx, cy = int(specific_face_landmark.x * w), int(specific_face_landmark.y * h)
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

                    if results.right_hand_landmarks:
                        r_specific_hand_landmark = results.right_hand_landmarks.landmark[specific_hand_landmark_id]

                        rx, ry = int(r_specific_hand_landmark.x * w), int(r_specific_hand_landmark.y * h)
                        cv2.circle(image, (rx, ry), 5, (0, 255, 0), -1)

                        r_hand_x, r_hand_y = int(r_specific_hand_landmark.x * w), int(r_specific_hand_landmark.y * h)
                        r_distance = np.linalg.norm(np.array((r_hand_x, r_hand_y)) - np.array((cx, cy)))
                        threshold = 20  # Adjust this value as needed

                        if r_distance <= threshold:

                            happyMascot = cv2.imread("statics/GoodJobMascotImage copy.jpeg")
                            x_offset = w - happyMascot.shape[1] - 10
                            y_offset = 10
                            image[y_offset:y_offset + happyMascot.shape[0],
                            x_offset:x_offset + happyMascot.shape[1]] = happyMascot
                            cv2.putText(image, f'Correct spot! Stay there for {processing_duration} seconds', (50, 80), font, 1,
                                        (41, 41, 41), 3, cv2.LINE_4)

                            if start_time is None:
                                start_time = time.time()
                            current_duration = time.time() - start_time
                            print(current_duration)
                            accurate_lap_times.append(current_duration+current_max)

                        else:
                            sadMascot = cv2.imread("statics/NotGoodJobMascotImage copy.jpeg")
                            x_offset = w - sadMascot.shape[1] - 10
                            y_offset = 10
                            image[y_offset:y_offset + sadMascot.shape[0],
                            x_offset:x_offset + sadMascot.shape[1]] = sadMascot
                            cv2.putText(image, f'OOPS, massage the blue circle with your right index finger!', (50, 80), font, 1, (41, 41, 41),
                                        3, cv2.LINE_4)
                        
                            if start_time is not None:
                                start_time = None
                                current_max = accurate_lap_times[-1]
                                accurate_lap_times.append(accurate_lap_times[-1])

            # Display current_max on the frame
            y_offset = 50
            cv2.putText(image, f'Accumulated Duration: {accurate_lap_times[-1]:.2f} seconds',
                        (10, 70 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(f"Detection ({processing_duration} seconds)", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Check if the total duration exceeds the processing_duration
            if accurate_lap_times[-1] >= processing_duration:
                processing += processing_duration
                break


    # Save the duration_list to CSV
    try:
        with open('time_tracker.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(accurate_lap_times)
    except FileNotFoundError:
        with open('time_tracker.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(accurate_lap_times)

    print(f"Durations for {processing_duration} seconds:", current_max)

# Display accurate lap times
print("Accurate Lap Times:", current_max) #accurate_lap_times[-1]

vidcap.release()
cv2.destroyAllWindows()
