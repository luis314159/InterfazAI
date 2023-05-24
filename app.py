from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
import base64

# Carga el modelo
model_path = ".\callback-001\callback"
model = load_model(model_path)

app = Flask(__name__)

# Reemplazar con tu modelo
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

"""def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
"""
@app.route('/')
def index():
    return render_template('index.html')

"""@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""



def procesar_prediccion(prediction):
    # Obtén el índice del valor más alto
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    if predicted_class == 0:
        return "grisley"
    if predicted_class == 1:
        return "panda"
    if predicted_class == 2:
        return "polar"
    
@app.route('/predict', methods=['POST'])
def predict():
    # Recupera los datos de la imagen del cuerpo de la solicitud
    image_data = request.get_data().decode('utf-8')
    
    # Si los datos de la imagen están en base64, elimina la parte de encabezado de la cadena
    if 'base64,' in image_data:
        image_data = image_data.split('base64,')[1]
    
    # Convierte los datos de la imagen de base64 a bytes
    image_data = base64.b64decode(image_data)
    
    # Abre la imagen con PIL
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Asegúrate de que tu imagen sea del tamaño correcto y esté normalizada como espera tu modelo
    image = image.resize((224, 224))
    image_np = np.array(image) / 255.0
    image_np = image_np.reshape(1,224,224,3)
    print(image_np.shape)

    # Asegúrate de que tu imagen tenga la forma correcta para tu modelo
    # Si tu modelo espera una lista de imágenes, es posible que necesites agregar una dimensión extra
    #image_np = np.expand_dims(image_np, axis=0)

    # Usar tu modelo para hacer una predicción
    prediction = model.predict(image_np)
    print(prediction)

    # Procesa la predicción para enviarla de vuelta al cliente
    # Esto dependerá de cómo quieras mostrar la predicción en el cliente
    prediction = procesar_prediccion(prediction)

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
