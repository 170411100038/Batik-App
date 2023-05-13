from flask import Flask, render_template, request, jsonify, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from praproses import impute_most_frequent, seleksi_fitur
from id3 import id3biasa, evaluate
import pandas as pd
import os
import json


app = Flask(__name__)
app.config["SECRET_KEY"] = "qwerty123"
app.config["UPLOAD_FOLDER"] = 'static'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")
    
class Pengujian(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

def Average(dat):
    return sum(dat) / len(dat)

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

@app.route("/praproses", methods=["POST", "GET"])
def praproses():
    form = UploadFileForm()
    pesan = "Silahkan upload file"
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        pesan = "File "+ str(file.filename) + " sudah terupload"
        
        # Data Awal
        target = os.path.join(app.static_folder, file.filename)
        data = pd.read_excel(target)
        # print(data)
        nama_kolom = data.columns.tolist()
        file_data = []
        for i in nama_kolom:
            file_data.append(data[i])
        # print (file_data[14][176:182])
        jumlah = [i for i in range(len(file_data[0]))]
        
        # Imputasi Data
        data_imputer = impute_most_frequent(data) # Memanggil fungsi imputasi dengan parameter data awal
        label = data_imputer['Label'] # Memindahkan kolom Label
        data_imputer = data_imputer.drop(['Label'], axis=1) # Drop kolom label
        data_imputer = data_imputer.astype('int') # Merapikan data dari 1.0 ke 1
        data_imputer_string = data_imputer.astype('string') # Merubah tipe data dari int ke string
        nama_kolom_imputasi = data_imputer.columns.tolist()
        
        file_data_imputer = []
        for j in nama_kolom_imputasi:
            file_data_imputer.append(data_imputer[j])
        
        # Seleksi Fitur
        info_gain = seleksi_fitur(data_imputer_string, label)
        banyak_fitur = 10
        data_seleksi = data_imputer[info_gain['value'][0:banyak_fitur]]
        nama_kolom_seleksi = data_seleksi.columns.tolist()
        data_seleksi['Label'] = label

        file_data_seleksi = []
        for k in nama_kolom_seleksi:
            file_data_seleksi.append(data_seleksi[k])
        
        return render_template("praproses.html", nama_kolom=nama_kolom, nama_kolom_imputasi=nama_kolom_imputasi, nama_kolom_seleksi=nama_kolom_seleksi, form=form, pesan=pesan, file_data=file_data, file_data_imputer=file_data_imputer, file_data_seleksi=file_data_seleksi, jumlah=jumlah)
    
    return render_template("praproses.html", form=form, pesan=pesan)

@app.route("/pengujian", methods=["POST", "GET"])
def pengujian():
    form = UploadFileForm()
    pesan = "Silahkan upload file"
    
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        pesan = "File "+ str(file.filename) + " sudah terupload"
        
        target = os.path.join(app.static_folder, file.filename)
        data = pd.read_excel(target)
        
        # Imputasi Data
        data_imputer = impute_most_frequent(data) # Memanggil fungsi imputasi dengan parameter data awal
        label = data_imputer['Label'] # Memindahkan kolom Label
        data_imputer = data_imputer.drop(['Label'], axis=1) # Drop kolom label
        data_imputer = data_imputer.astype('int') # Merapikan data dari 1.0 ke 1
        data_imputer_string = data_imputer.astype('string') # Merubah tipe data dari int ke string
        
        info_gain = seleksi_fitur(data_imputer_string, label)
        banyak_fitur = 10
        data_seleksi = data_imputer[info_gain['value'][0:banyak_fitur]]
        nama_kolom_seleksi = data_seleksi.columns.tolist()
        data_seleksi['Label'] = label

        file_data_seleksi = []
        for k in nama_kolom_seleksi:
            file_data_seleksi.append(data_seleksi[k])
        
        jumlah = [i for i in range(len(file_data_seleksi[0]))]
        
        akurasi, presisi, recall, Test, Train = id3biasa(data_seleksi, 5)
        
        data_test = []
        for j in nama_kolom_seleksi:
            data_test.append(Test[j])
        jumlah_test = [i for i in range(len(data_test[0]))]
        
        data_train = []
        for i in nama_kolom_seleksi:
            data_train.append(Train[i])
        jumlah_train = [i for i in range(len(data_train[0]))]
        
        return render_template("pengujian.html", nama_kolom_seleksi=nama_kolom_seleksi, form=form, pesan=pesan, file_data_seleksi=file_data_seleksi, jumlah=jumlah, akurasi=akurasi, presisi=presisi, recall=recall, data_test=data_test, jumlah_test=jumlah_test, data_train=data_train, jumlah_train=jumlah_train)
    
    return render_template("pengujian.html", form=form, pesan=pesan)

@app.route("/uji", methods=["POST", "GET"])
def uji():
    if request.method == "POST" :
        # Opening JSON file
        f = open('Tree_ID3.json')
        tree = json.load(f)
        
        Pregnancies = pd.DataFrame([request.form["Pregnancies"]], columns=['Pregnancies'])
        Glucose = pd.DataFrame([request.form["Glucose"]], columns=['Glucose'])
        BloodPressure = pd.DataFrame([request.form["BloodPressure"]], columns=['BloodPressure'])
        SkinThickness = pd.DataFrame([request.form["SkinThickness"]], columns=['SkinThickness'])
        Insulin = pd.DataFrame([request.form["Insulin"]], columns=['Insulin'])
        BMI = pd.DataFrame([request.form["BMI"]], columns=['BMI'])
        DiabetesPedigreeFunction = pd.DataFrame([request.form["DiabetesPedigreeFunction"]], columns=['DiabetesPedigreeFunction'])
        Age = pd.DataFrame([request.form["Age"]], columns=['Age'])
        Outcome = pd.DataFrame(["C1"], columns=['Outcome'])
        
        dataInput = pd.concat([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome], axis=1)
        
        dataAwal = pd.read_excel("Batik.xlsx")
        
        # Imputasi Data
        data_imputer = impute_most_frequent(dataAwal) # Memanggil fungsi imputasi dengan parameter data awal
        label = data_imputer['Label'] # Memindahkan kolom Label
        data_imputer = data_imputer.drop(['Label'], axis=1) # Drop kolom label
        data_imputer = data_imputer.astype('int') # Merapikan data dari 1.0 ke 1
        data_imputer_string = data_imputer.astype('string') # Merubah tipe data dari int ke string
        
        info_gain = seleksi_fitur(data_imputer_string, label)
        banyak_fitur = 10
        data_seleksi = data_imputer[info_gain['value'][0:banyak_fitur]]
        data_seleksi['Label'] = label
        
        dataGabung = pd.concat([dataAwal, dataInput], ignore_index=True)
        dataInput2 = dataGabung.tail(1)
        
        hasil_input = evaluate(tree, dataInput2)
        
        return render_template("uji.html", hasil_input=hasil_input)
    
    return render_template("uji.html")

@app.route("/evaluasi")
def evaluasi():
    data = pd.read_excel("output_ID3_imputasi.xlsx")
    akurasi = data["Akurasi"]
    presisi = data["Presisi"]
    recall = data["Recall"]
    dat_data = [akurasi, presisi, recall]
    jumlah = [i for i in range(len(data))]
    print("Jumlah :", jumlah)
    rat_akurasi = round(Average(akurasi),2)
    rat_presisi = round(Average(presisi),2)
    rat_recall = round(Average(recall),2)
    rat_data = [rat_akurasi, rat_presisi, rat_recall]
    
    return render_template("evaluasi.html", rat_data=rat_data, dat_data=dat_data, jumlah=jumlah)

@app.route("/about")
def redirect_about():
    return "Aboutt"

if __name__ == "__main__":
    app.run(debug=True)