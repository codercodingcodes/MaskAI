from tensorflow import keras
import numpy as np
import math
import json
import requests
import cv2
import time
from flask import Flask, render_template, Response, request,jsonify
from face_detect import face_recong
# --load test and training data--
# data = np.load("data.npy")
# labels = np.load("label.npy")
# test_len = int(len(data)*0.15)
# data_len = len(data)
# print(data_len)
# print(test_len)

# this is in mm
pix_size = 0.26
head_height = 225

# this is in cubic meter
tidal_vol = 0.0005
speech_vol = 0.001
# this is in seconds
avg_breath = 5
# key
key ='ec81b031f4ab416298231710212003'
location = input("first three letter of postal code?")
window_dim = float(input("Dimension of windows in feet?"))
window_dir = 270 - int(input("Direction of windows due north?"))

payload = {
    "key":key,
    'q':location,
}
r = requests.get('http://api.weatherapi.com/v1/current.json', params=payload)
if r.status_code != 200:
    print("error api "+str(r.status_code))
    quit()
weather_data = json.dumps(r.json(),sort_keys=True,indent=4)
print(weather_data)

p=0
override = ""

if override == "yes":
    quit()
    # -- train model --
    # model = keras.Sequential([
    #     keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', input_shape=shape),
    #     keras.layers.Conv1D(32, kernel_size=3, strides=1, activation='relu', input_shape=shape),
    #     keras.layers.MaxPooling1D(pool_size=2),
    #
    #     keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', input_shape=shape),
    #     keras.layers.Conv1D(32, kernel_size=3, strides=1, activation='relu', input_shape=shape),
    #     keras.layers.MaxPooling1D(pool_size=2),
    #     keras.layers.Dropout(0.2),
    #
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(100,activation="relu"),
    #     keras.layers.Dense(70,activation="relu"),
    #     keras.layers.Dense(50,activation="relu"),
    #     keras.layers.Dense(2,activation="softmax"),
    # ])
    #
    # model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    # model.fit(training_data,training_label,epochs=10,batch_size=5,shuffle=True)
    # results = model.evaluate(testing_data, testing_label)
    # model.save("model.h5")
else:

    model = keras.models.load_model("model.h5")

    def translate(image):

        width = 100
        height = 100

        resized_img = cv2.resize(image, (width, height))
        greyscale = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(greyscale, (5, 5), 0)
        return blur / 255
    stream = cv2.VideoCapture(0)

    def frame_process():
        global p
        count = 0
        cooldown = 0
        Kw = 0.3
        # for calculating airflow per second
        if (int(r.json()['current']['wind_degree']) - window_dir) >= 150 and (int(r.json()['current']['wind_degree']) - window_dir) <= 210:
            Kw = 0.5

        if r.json()['current']['wind_mph']>0:
            flow = (88 * Kw * window_dim * float(r.json()['current']['wind_mph'])) / (35.315 * 60)
        else:
            flow = 0.000000001
        print(flow)
        while True:
            masked = 0
            unmasked = 0

            success, frame = stream.read()
            face = face_recong(frame)
            all_faces = face[0]
            face_img = frame
            for (x, y, w, h) in face[1]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            predictions = []

            for face in all_faces:
                image = np.array(translate(face)).reshape(1,100,100)
                prediction = model.predict(image)
                if np.argmax(prediction[0]) >= 1:
                    predictions.append("on")
                    masked += 1
                else:
                    predictions.append("off")
                    unmasked += 1
            print(predictions)
            if unmasked > 0:
                count += 0.5
                cooldown = 0
            elif cooldown > 15:
                count = 0
            else:
                cooldown += 1

            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', face_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            p = 1 - math.exp(-(unmasked*(tidal_vol/avg_breath)*count*48)/flow)
            print(count)
            print(p*100)
            print(unmasked)
            time.sleep(0.5)

    app = Flask(__name__)
    @app.route("/video")
    def video():
        return Response(frame_process(),mimetype='multipart/x-mixed-replace; boundary=frame')



    @app.route('/')
    def index():
        return render_template('Main.html')

    @app.route("/test",methods=['GET','POST'])
    def test():
        if request.method == 'GET':
            message = {'p':str(p*100)}
            return jsonify(message)
    if __name__ == "__main__":
        app.run(debug=False)







