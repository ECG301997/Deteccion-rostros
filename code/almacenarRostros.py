'''
Programa para detectar y almacenar Rostros
'''
import cv2
import os

imagesPath = "./imagenesRostros"
imagesPathList = os.listdir(imagesPath)

if not os.path.exists('./imagenesRostros'):
    os.makedirs('./imagenesRostros')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

for imageName in imagesPathList:
    image = cv2.imread(imagesPath+'/'+imageName)
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(image, 'Presione s, para almacenar los rostros encontrados', (10,20), 2, 0.7, (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow('image', image)
    k = cv2.waitKey(0)
    if k == ord('s'):
        for (x,y,w,h) in faces:
            rostro = imageAux[y:y+h,x:x+w]
            rostro = cv2.resize(rostro, (150,150), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite('imagenesRostros/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
    elif k==27:
        break
cv2.destroyAllWindows()