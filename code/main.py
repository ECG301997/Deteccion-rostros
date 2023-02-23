import detectorPOO as detect
import cv2

def main():
    detector = detect.DetectorRostros()
    nombre = './code/oficina.jpeg'
    imagen = detector.detectar(nombre)
    cv2.imshow('deteccion POO',imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()