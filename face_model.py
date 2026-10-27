import cv2
import os
import numpy as np

FOLDER = "known_faces"
os.makedirs(FOLDER, exist_ok=True)

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def load_faces():
    images = []
    labels = []
    names = {}

    for name in os.listdir(FOLDER):
        person_folder = os.path.join(FOLDER, name)

        if not os.path.isdir(person_folder):
            continue

        label = len(names)
        count = 0

        for file in os.listdir(person_folder):
            path = os.path.join(person_folder, file)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                images.append(cv2.resize(image, (200, 200)))
                labels.append(label)
                count += 1

        if count:
            names[label] = name

    return images, np.array(labels), names


def register():
    name = input("Enter person's name: ").strip()

    if not name or any(c in name for c in r'<>:"/\|?*'):
        print("Please enter a valid name.")
        return

    person_folder = os.path.join(FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)

    camera = cv2.VideoCapture(0)
    saved = 0

    print("Look at the camera.")
    print("Press S to save a face; press Q when finished.")
    print("Capture around 15-20 images with slight pose changes.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Cannot read webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(80, 80)
        )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)

        cv2.putText(frame, f"Saved: {saved}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                path = os.path.join(
                    person_folder, f"{saved}.jpg"
                )
                cv2.imwrite(path, face)
                saved += 1
                print("Saved image:", saved)
            else:
                print("Ensure exactly one face is visible.")

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Registration finished. Images saved:", saved)


def recognize():
    images, labels, names = load_faces()

    if len(images) == 0:
        print("No registered faces. Choose option 1 first.")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    print("Live recognition started. Press Q to exit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(80, 80)
        )

        for x, y, w, h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, confidence = model.predict(face)

            if label in names and confidence < 65:
                name = names[label]
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)
            cv2.putText(frame, name, (x, max(y-10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


print("\n=== LIVE FACE RECOGNITION ===")
print("1. Register a person")
print("2. Start live recognition")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
    register()
elif choice == "2":
    recognize()
else:
    print("Invalid choice.")