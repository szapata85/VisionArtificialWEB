#importar librerias
from flask import  Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import time


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
    #variables
    parpadeo = False
    conteo = 0
    tiempo = 0
    inicio = 0
    final = 0
    conteo_sueños = 0
    muestra = 0


    #Empezamos
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        #Listas
        px = []
        py = []
        lista = []


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

                    for id, puntos in enumerate(rostros.landmark):
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)

                        lista.append([id, x, y])

                        if len(lista) == 468:
                            #ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]

                            longitud1 = math.hypot(x2-x1, y2-y1)

                            #ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]

                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            cv2.putText(frame, f'Parpadeos: {int(conteo)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f'Micro Suenos: {int(conteo_sueños)}', (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame, f'Duracion: {int(muestra)}', (210, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                            if longitud1 <= 10 and longitud2 <= 10 and not parpadeo:
                                conteo = conteo + 1
                                parpadeo = True
                                inicio = time.time()
                            elif longitud2 > 10 and longitud1 > 10 and parpadeo:
                                parpadeo = False
                                final = time.time()

                            tiempo = round(final - inicio, 0)

                            if tiempo >= 3:
                                conteo_sueños = conteo_sueños + 1
                                muestra = tiempo
                                inicio  = 0
                                final = 0




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



