from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import logging
from prediction import predict

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # get image from request
        img = request.files['img'].stream
        img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # make prediction
        pred_label, pred_conf = predict(img)

        response = {
            'label': str(pred_label),
            'confidence': str(pred_conf)
        }
        
        return jsonify(response)


@app.route('/upload-ktp', methods=['POST'])
def upload_ktp():
    if request.method == 'POST':
        response = {
            'nik': '3507322922020012',
            'nama' : 'Wiradarma Nurmagika Bagaskara',
            'ttl' : 'Malang, 29 Desember 2002',
            'jenis_kelamin' : 'Laki-laki',
            'alamat' : 'Jl. Kawi No. 1, Malang',
            'agama' : 'Islam',
            'status_perkawinan' : 'Belum Kawin',
            'pekerjaan' : 'Pelajar/Mahasiswa',
            'kewarganegaraan' : 'WNI',
            'berlaku_hingga' : 'Seumur Hidup'
        }
        
        return jsonify(response)