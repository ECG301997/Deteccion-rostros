import cv2

# programa para detectar los rostros del archivo rostros.jpeg

# Cargar el clasificador cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen

img = cv2.imread('./code/oficina.jpeg')

# Convertir a escala de grises

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30,30),
                                     maxSize=(200,200))


# Dibujar rectangulos alrededor de los rostros

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#Mostrar la imagen
cv2.imshow('deteccion 1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()