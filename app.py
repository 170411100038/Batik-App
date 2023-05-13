from flask import Flask, render_template, request, jsonify, flash
from praproses import impute_most_frequent, seleksi_fitur
from id3 import id3biasa, evaluate, prediksi_bagging
import pandas as pd
import os
import json


app = Flask(__name__)
app.config["SECRET_KEY"] = "qwerty123"
app.config["UPLOAD_FOLDER"] = 'static'

def Average(dat):
    return sum(dat) / len(dat)

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

@app.route("/praproses", methods=["POST", "GET"])
def praproses():    
    # Data Awal
    data = pd.read_excel("Batik.xlsx")
    
    cek_nan = data.copy()
    cek_nan.fillna("NaN", inplace=True)
    # print(data)
    nama_kolom = cek_nan.columns.tolist()
    file_data = []
    for i in nama_kolom:
        file_data.append(cek_nan[i])
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
    
    return render_template("praproses.html", nama_kolom=nama_kolom, nama_kolom_imputasi=nama_kolom_imputasi, nama_kolom_seleksi=nama_kolom_seleksi, file_data=file_data, file_data_imputer=file_data_imputer, file_data_seleksi=file_data_seleksi, jumlah=jumlah)
    
    # return render_template("praproses.html", form=form, pesan=pesan)

@app.route("/pengujian", methods=["POST", "GET"])
def pengujian():
    data = pd.read_excel("Batik.xlsx")
    
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
    
    return render_template("pengujian.html", nama_kolom_seleksi=nama_kolom_seleksi, file_data_seleksi=file_data_seleksi, jumlah=jumlah, akurasi=akurasi, presisi=presisi, recall=recall, data_test=data_test, jumlah_test=jumlah_test, data_train=data_train, jumlah_train=jumlah_train)
    
    # return render_template("pengujian.html", form=form, pesan=pesan)

@app.route("/uji", methods=["POST", "GET"])
def uji():
    if request.method == "POST" :
        # Opening JSON file
        f = open('Tree_F2_K15.json')
        tree = json.load(f)
        print(tree[0])
        # tree = list(tree.values())
        
        Jml_Karyawan = pd.DataFrame([request.form["Jumlah karyawan"]], columns=['jumlah karyawan'])
        FP_Online = pd.DataFrame([request.form["Fasilitas pembayaran online"]], columns=['fasilitas pembayaran online'])
        M_S_Ijin_Usaha = pd.DataFrame([request.form["Memiliki surat ijin usaha"]], columns=['memiliki surat ijin usaha'])
        P_Bersertifikat_IT = pd.DataFrame([request.form["Pegawai bersertifikat IT"]], columns=['pegawai bersertifikat IT'])
        SI_P_Batik_Sendiri = pd.DataFrame([request.form["SI pengelolaan batik sendiri"]], columns=['SI pengelolaan batik sendiri'])
        Branding_P = pd.DataFrame([request.form["Terdapat branding produk"]], columns=['terdapat branding produk'])
        M_Marketplace = pd.DataFrame([request.form["Mempunyai marketplace"]], columns=['mempunyai marketplace'])
        Aturan_PB_Offline = pd.DataFrame([request.form["Aturan pembelian batik offline"]], columns=['aturan pembelian batik offline'])
        F_P_Cvd19 = pd.DataFrame([request.form["Fasilitas pencegahan covid-19"]], columns=['fasilitas pencegahan covid-19'])
        K_D_Mitra = pd.DataFrame([request.form["Kerjasama dengan mitra"]], columns=['kerjasama dengan mitra'])
        Label = pd.DataFrame(["C1"], columns=['Label'])
        
        dataInput = pd.concat([Jml_Karyawan, FP_Online, M_S_Ijin_Usaha, P_Bersertifikat_IT, SI_P_Batik_Sendiri, Branding_P, M_Marketplace, Aturan_PB_Offline, F_P_Cvd19, K_D_Mitra, Label], axis=1)
        # print(dataInput)
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
        
        dataGabung = pd.concat([data_seleksi, dataInput], ignore_index=True)
        dataInput2 = dataGabung.tail(1)
        
        out_label = dataInput2['Label']
        dataInput2 = dataInput2.drop(['Label'], axis=1) # Drop kolom label
        dataInput2 = dataInput2.astype('float')
        dataInput2 = dataInput2.astype('string')
        dataInput2['Label'] = out_label

        print(dataInput2)
        
        hasil_input = prediksi_bagging(tree, dataInput2)
        
        return render_template("uji.html", hasil_input=hasil_input)
    
    return render_template("uji.html")

@app.route("/evaluasi")
def evaluasi():
    data1 = pd.read_excel("ID3_bagging_imputasi_akurasi_KNN_seleksi.xlsx")
    data2 = pd.read_excel("ID3_bagging_imputasi_akurasi_modus_seleksi.xlsx")
    data3 = pd.read_excel("C45_bagging_imputasi_akurasi_KNN_seleksi.xlsx")
    data4 = pd.read_excel("C45_bagging_imputasi_akurasi_modus_seleksi.xlsx")
    k = 14
    
    # ID3 Bagging
    data_id3_knn = []
    for i in range(k):
        data_id3_knn.append(data1['k:'+str(i+2)])
    jumlah = [i for i in range(len(data_id3_knn[0]))]
    rat_akurasi_id3_knn = []
    for j in range(k):
        rat_akurasi_id3_knn.append(round(Average(data_id3_knn[j]),2))
        
    data_id3_modus = []
    for i in range(k):
        data_id3_modus.append(data2['k:'+str(i+2)])
    rat_akurasi_id3_modus = []
    for j in range(k):
        rat_akurasi_id3_modus.append(round(Average(data_id3_knn[j]),2))
        
    # C45 Bagging
    data_c45_knn = []
    for i in range(k):
        data_c45_knn.append(data3['k:'+str(i+2)])
    rat_akurasi_c45_knn = []
    for j in range(k):
        rat_akurasi_c45_knn.append(round(Average(data_id3_knn[j]),2))
        
    data_c45_modus = []
    for i in range(k):
        data_c45_modus.append(data4['k:'+str(i+2)])
    rat_akurasi_c45_modus = []
    for j in range(k):
        rat_akurasi_c45_modus.append(round(Average(data_id3_knn[j]),2))
    
    return render_template("evaluasi.html",k=k,data_c45_knn=data_c45_knn, data_c45_modus=data_c45_modus, data_id3_knn=data_id3_knn, data_id3_modus=data_id3_modus, rat_akurasi_id3_knn=rat_akurasi_id3_knn, rat_akurasi_id3_modus=rat_akurasi_id3_modus, rat_akurasi_c45_knn=rat_akurasi_c45_knn, rat_akurasi_c45_modus=rat_akurasi_c45_modus, jumlah=jumlah)

@app.route("/about")
def redirect_about():
    return "Aboutt"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
