#importar librerias
from flask import  Flask, render_template, Response
import cv2
import mediapipe as mp


#Creamos nuestra funcion de Dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibujo = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

#Creamos un objeto donde almacenamos la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

#Realizar videoCaptura
cap = cv2.VideoCapture(0)

#Funcion Frames
def genframe():
    #Empezamos
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        else:
            #Correccion de color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Observamos los resultados
            resultados = MallaFacial.process(frameRGB)

            #Si tenemos rostro
            if resultados.multi_face_landmarks:
                #Iteramos
                for rostros in resultados.multi_face_landmarks:
                    #Dibujamos
                    mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibujo, ConfDibujo)

            suc, encode = cv2.imencode('.jpg', frame)
            frame = encode.tobytes()

        yield(b'--frame\r\n'
              b'content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Creamos la App
app = Flask(__name__)

#Ruta Principal
@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/video')
def video():
    return Response(genframe(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)



