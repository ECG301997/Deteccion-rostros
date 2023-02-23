import cv2

# programa para detectar rostros y a travez del control de barra deslizante otroga niveles de desenfoque
# ademas se puede mostrar la imagen a color o escala de grises

def nothing(x):
    pass

# leer la imagen a color
img = cv2.imread("./code/oficina.jpeg")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# especificar el nombre de la ventana
cv2.namedWindow('rostro')

# Crear los trackbars para el control de la imagen
cv2.createTrackbar('Blur','rostro',0,15,nothing)
cv2.createTrackbar('Gray','rostro',0,1,nothing)

# Obtener las posiciones de las barras de seguimiento
while True:
    val = cv2.getTrackbarPos('Blur','rostro')
    grayVal = cv2.getTrackbarPos('Gray','rostro')
  
    # condicion de escala de grises o colores
    if grayVal == 1:
        imagenN = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    else:
        imagenN = img.copy()
        
    # Detectar rostros y difuminarlos a diferentes niveles
    faces = face_cascade.detectMultiScale(imagenN,1.1,5)

    for (x,y,w,h) in faces:
        if val > 0:
            imagenN[y:y+h,x:x+w] = cv2.blur(imagenN[y:y+h, x:x+w], (val,val))
    
    # Mostrar imagen
    cv2.imshow('rostro',imagenN)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        

cv2.destroyAllWindows()
