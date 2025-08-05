import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

gesture_name = 'thumbs_up'
save_path = f"data/{gesture_name}"
os.makedirs(save_path, exist_ok=True)
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    roi = frame[100:300, 100:300]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    if count < 300:
        cv2.imwrite(f"{save_path}/{count}.jpg", roi)
        count += 1
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

data = []
labels = []
classes = ['thumbs_up', 'thumbs_down', 'ok', 'peace']
for idx, gesture in enumerate(classes):
    folder = f"data/{gesture}"
    for file in os.listdir(folder):
        img = load_img(os.path.join(folder, file), target_size=(64, 64), color_mode="grayscale")
        img_array = img_to_array(img) / 255.0
        data.append(img_array)
        labels.append(idx)
data = np.array(data)
labels = to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(classes), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=32)
model.save("gesture_model.h5")

model = load_model("gesture_model.h5")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    roi = frame[100:300, 100:300]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = roi.reshape(1, 64, 64, 1) / 255.0
    pred = model.predict(roi)
    class_idx = np.argmax(pred)
    class_label = classes[class_idx]
    cv2.putText(frame, class_label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
