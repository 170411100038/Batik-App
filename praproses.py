from sklearn.impute import SimpleImputer
import pandas as pd

def impute_most_frequent(data):
    imputer = SimpleImputer(strategy='most_frequent')
    
    data['kenaikan harga bahan baku'] = imputer.fit_transform(data['kenaikan harga bahan baku'].values.reshape(-1,1))[:,0]
    data['pelatihan pemilik pertahun'] = imputer.fit_transform(data['pelatihan pemilik pertahun'].values.reshape(-1,1))[:,0]
    data['jumlah variasi motif batik'] = imputer.fit_transform(data['jumlah variasi motif batik'].values.reshape(-1,1))[:,0]
    data['menerapkan new normal'] = imputer.fit_transform(data['menerapkan new normal'].values.reshape(-1,1))[:,0]
    data['aturan pembelian batik offline'] = imputer.fit_transform(data['aturan pembelian batik offline'].values.reshape(-1,1))[:,0]
    data['fasilitas pencegahan covid-19'] = imputer.fit_transform(data['fasilitas pencegahan covid-19'].values.reshape(-1,1))[:,0]
    data['pegawai bersertifikat IT'] = imputer.fit_transform(data['pegawai bersertifikat IT'].values.reshape(-1,1))[:,0]
    data['pendidikan pemilik'] = imputer.fit_transform(data['pendidikan pemilik'].values.reshape(-1,1))[:,0]
    data['mempunyai marketplace'] = imputer.fit_transform(data['mempunyai marketplace'].values.reshape(-1,1))[:,0]
    data['fasilitas pembayaran online'] = imputer.fit_transform(data['fasilitas pembayaran online'].values.reshape(-1,1))[:,0]
    data['SI pengelolaan batik sendiri'] = imputer.fit_transform(data['SI pengelolaan batik sendiri'].values.reshape(-1,1))[:,0]
    data['jumlah karyawan'] = imputer.fit_transform(data['jumlah karyawan'].values.reshape(-1,1))[:,0]
    
    return data

from sklearn.feature_selection import mutual_info_classif

def seleksi_fitur(data, label):
    IG = list(mutual_info_classif(data, label, discrete_features=True))

    column_headers = list(data.columns.values)
    IG_S = pd.DataFrame(column_headers, columns = ['value'])
    IG_S['IG'] = IG
    IG_S = IG_S.sort_values(by=['IG'], ascending=False)
    
    return IG_S