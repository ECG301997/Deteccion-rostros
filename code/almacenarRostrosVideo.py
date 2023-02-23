import cv2
import os

if not os.path.exists('./imagenesRostrosVideo'):
    os.makedirs('./imagenesRostrosVideo')

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.1, 5)

    k = cv2.waitKey(1)
    if k == 27:
        break

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation = cv2.INTER_CUBIC)
        if k == ord('s'):
            cv2.imwrite('imagenesRostrosVideo/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
    cv2.rectangle(frame, (10,5), (550,65), (0,0,0), -1)
    cv2.putText(frame, 'Presione s, para almacenar los rostros encontrados', (10,20), 2, 0.7, (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()