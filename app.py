import os
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from flask import Flask, render_template, Response, request, url_for
from twilio.rest import Client

app = Flask(__name__)

# ---------------- FOLDER SETUP ----------------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- TWILIO CONFIG ----------------
TWILIO_ACCOUNT_SID = "AC7b631c3ec0b1484453aa3969a5b445df"
TWILIO_AUTH_TOKEN = "04135df35e9ae1e5b5c73e85c47e9cb6"
TWILIO_PHONE = "+17657802495"
MY_PHONE = "+916383568873"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms_alert():
    try:
        twilio_client.messages.create(
            body="🚨 Driver Drowsiness Detected! Please check immediately.",
            from_=TWILIO_PHONE,
            to=MY_PHONE
        )
        print("SMS Sent Successfully")
    except Exception as e:
        print("SMS Error:", e)

# ---------------- MEDIAPIPE SETUP ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
YAWN_POINTS = [13, 14]

# ---------------- GLOBAL VARIABLES ----------------
drowsy_start_time = None
blink_count = 0
blink_ready = True
sms_sent = False

# ---------------- FUNCTIONS ----------------

def eye_aspect_ratio(eye_indices, lm):
    try:
        p = [lm[i] for i in eye_indices]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C)
    except:
        return 0.3

def detect_emotion(lm):
    mouth_h = np.linalg.norm(lm[13] - lm[14])
    mouth_w = np.linalg.norm(lm[61] - lm[291])
    if mouth_h > 25:
        return "SURPRISED 😲"
    if mouth_w > 55 and mouth_h < 15:
        return "HAPPY 😊"
    return "NEUTRAL 😐"

def process_frame_logic(frame):
    global drowsy_start_time, blink_count, blink_ready, sms_sent

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        for face_landmarks in res.multi_face_landmarks:
            lm = np.array([(int(p.x*w), int(p.y*h)) for p in face_landmarks.landmark])

            # Draw Eye Points
            for idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, lm[idx], 2, (0, 255, 0), -1)

            # EAR
            l_ear = eye_aspect_ratio(LEFT_EYE, lm)
            r_ear = eye_aspect_ratio(RIGHT_EYE, lm)
            ear = (l_ear + r_ear) / 2.0

            # Blink & Drowsy
            if ear < 0.21:
                if blink_ready:
                    blink_count += 1
                    blink_ready = False

                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif (time.time() - drowsy_start_time) >= 3.0:
                    cv2.putText(frame, "!!! DROWSY ALERT !!!",
                                (w//4, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 4)

                    if not sms_sent:
                        threading.Thread(target=send_sms_alert).start()
                        sms_sent = True
            else:
                drowsy_start_time = None
                blink_ready = True
                sms_sent = False

            # Yawn
            mouth_distance = np.linalg.norm(lm[YAWN_POINTS[0]] - lm[YAWN_POINTS[1]])
            if mouth_distance > 30:
                cv2.putText(frame, "YAWN WARNING!",
                            (w//2-120, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 165, 255), 3)

            # Emotion
            emotion = detect_emotion(lm)

            # UI Text
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # EAR Bar
            bar_h = int(np.interp(ear, [0.1, 0.35], [0, 100]))
            cv2.rectangle(frame, (w-40, 150), (w-20, 50), (255,255,255), 1)
            cv2.rectangle(frame, (w-40, 150), (w-20, 150-bar_h), (0,255,0), -1)

    return frame

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_analysis', methods=['GET', 'POST'])
def image_analysis():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            img = cv2.imread(path)
            processed = process_frame_logic(img)

            res_name = "res_" + file.filename
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], res_name), processed)

            return render_template('image.html',
                                   result_img=url_for('static',
                                   filename='uploads/'+res_name))
    return render_template('image.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_analysis', methods=['GET', 'POST'])
def video_analysis():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            return render_template('video.html', video_file=file.filename)
    return render_template('video.html')

def gen_frames(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame_logic(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed/<source_type>/<filename>')
def video_feed(source_type, filename):
    source = 0 if source_type == 'live' else os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(gen_frames(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)