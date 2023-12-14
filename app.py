from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import logging
from prediction import predict
from firebase_admin import credentials, firestore, initialize_app
from spk import averageValue, pda, nda, sp, sn, nsp, nsn, ranking

app = Flask(__name__, template_folder='templates')

# initialize firestore 
cred = credentials.Certificate("./scan-sense-d7ad9-firebase-adminsdk-o1drh-6d4ddc5cc6.json")
default_app = initialize_app(cred)
db = firestore.client()
user_ref = db.collection('test')

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

@app.route('/users', methods=['GET'])
def read():
    """
        read() : Fetches documents from Firestore collection as JSON
        todo : Return document that matches query ID
        all_todos : Return all documents
    """
    try:
        # bidang 
        bidang = request.args.get('bidang')
        results = [doc.to_dict() for doc in user_ref.stream()]
        criterias = []
        alternative = []
        for user in results :
            data = user['kriteria']
            data["name"] = user["name"]
            criterias.append(data)
            alternative.append(user["name"])
        
        # average value 
        # sudah valid 
        avg = averageValue(criterias)

        # Menentukan jarak positif 
        # sudah valid 
        result_pda = pda(criterias, avg)

        # Menentukan jarak negatif 
        # sudah valid 
        result_nda = nda(criterias, avg)

        # Menentukan jumlah terbobot positif 
        # sudah valid 
        result_sp = sp(result_pda, "IT")

        # Menentukan jumlah terbobot negatif 
        # sudah valid 
        result_sn = sn(result_nda, "IT")

        # normaisasi sp
        # sudah valid 
        result_nsp = nsp(result_sp)

        # normalisasi sn 
        # sudah valid 
        result_nsn = nsn(result_sn)

        # perankingan 
        final = ranking(result_nsp, result_nsn, alternative)
        
        # print(results)
        return jsonify(final), 200
    except Exception as e:
        return f"An Error Occured: {e}"

