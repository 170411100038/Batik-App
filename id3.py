import pandas as pd #untuk memanipulasi data csv
import numpy as np #untuk perhitungan matematika
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report #untuk menampilkan performa dari sistem yang dibuat
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

def calc_total_entropy(train_data, label, kelas):
    total_baris = train_data.shape[0] #ukuran total dari dataset
    total_entropy = 0
    
    #pengulangan tiap kelas 0 dan 1
    for c in kelas: 
        #jumlah data dari kelas
        total_kelas = train_data[train_data[label] == c].shape[0] 
        #menghitung entropy tiap kelas
        prob = total_kelas/total_baris
        
        np.seterr(divide = 'ignore')
        log = np.where(prob>0, np.log2(prob), 0)
        # log = np.log2(prob)
        total_entropy_kelas = -prob * log
        #menambah var class entropy ke var total entropy dari dataset
        total_entropy += total_entropy_kelas 
    
    return total_entropy #mengembalikan nilai total entropy

def calc_entropy(feature_value_data, label, class_list):
    #ukuran dari tiap isi fitur misalkan jumlah value "rendah" adalah 100
    class_count = feature_value_data.shape[0] 
    entropy = 0
    
    for c in class_list:
        #jumlah data tiap kelas (0 & 1)
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] 
        entropy_class = 0
        #mengecek apakah jumlah data di kelas sekarang 0 atau tidak
        if label_class_count != 0: 
            #probabilitas tiap kelas dari tiap value fitur
            probability_class = label_class_count/class_count 
            #menghitung entropy
            entropy_class = - probability_class * np.log2(probability_class) 
        #menambahkan entropy dari tiap kelas ke variabel entropy
        entropy += entropy_class 
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    # mendapatkan isi dari fitur (ex: rendah, normal, tinggi)
    feature_value_list = train_data[feature_name].unique() 
    # mendapatkan total data
    total_row = train_data.shape[0] 
    feature_info = 0.0
    
    # perulangan tiap isi dari fitur
    for feature_value in feature_value_list:
        # memfilter data berdasarkan isi fitur yang looping sekarang
        feature_value_data = train_data[train_data[feature_name] == feature_value] 
        # mendapatkan ukuran dari isi fitur
        feature_value_count = feature_value_data.shape[0] 
        # menghitung entropy dari tiap isi fitur misal Pregnancies (ex: Rendah) entropynya berapa dst
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        # probabilitas dari total isi fitur dengan total keseluruhan data 
        feature_value_probability = feature_value_count/total_row
        # menghitung nilai informasi dari tiap isi fitur dan memasukannya ke dalam variabel feature_info
        feature_info += feature_value_probability * feature_value_entropy 
    
    #menghitung information gain dengan cara total entropy dikurangi nilai feature_info
    return calc_total_entropy(train_data, label, class_list) - feature_info 

def find_most_informative_feature(train_data, label, class_list):
    #mendapatkan nama tiap fitur dalam dataset
    feature_list = train_data.columns.drop(label) 
    #N.B. label bukan merupakan fitur, maka kita drop

    max_info_gain = -1
    max_info_feature = None
    
    #looping tiap fitur dalam dataset
    for feature in feature_list: 
        # calc_info_gain("Pregnancies", data training, "Outcome", [0,1])
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        # print("Information Gain dari", feature, ":", feature_info_gain)
        # pengecekan apakah info gain bernilai 0
        if feature_info_gain == 0 : 
            # jika iya maka nilai info gain akan diganti menjadi -1 sehingga tidak menjadi info gain terbesar
            feature_info_gain = -1 
        # memilih nilai information gain terbesar
        if max_info_gain < feature_info_gain: 
            max_info_gain = feature_info_gain
            max_info_feature = feature
    # print("Max info gain :", max_info_feature)
    # print("")
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list, cek=None):
    #mengecek apakah fitur yang sekarang di cek berisi None
    if feature_name==None:
        feature_name=cek
        #print(cek)
    # print("================================")
    # print("--> generate_sub_tree <--")
    # print("label :",label)
    # print("class_list :",class_list)

    #mendapatkan jumlah tiap isi dari fitur (ex: glucose rendah 5, normal 10, tinggi 5)
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) 
    # print("feature_value_count_dict :\n", feature_value_count_dict)
    # print("")
    
    f_left=10
    l_node=False
    
    cf= train_data.columns.values.tolist()
    #cek sisa fitur
    for f in cf: 
        #print(train_data[f].unique(),f)
        panj = len(train_data[f].unique())
        if panj == 1:
            f_left-=1
    if f_left<=1:
        l_node=True
    # print("cek sisa fitur :",l_node,f_left)

    tree = {} #tree
    #print("Tree :\n", tree)
    cek_fitur = cek
    # print("Fitur sebelumnya :", cek_fitur)

    #melakukan perulangan dari tiap isi dari fitur (ex : rendah 5, normal 10, tinggi 5)
    for feature_value, count in feature_value_count_dict.iteritems(): 
        # print("")
        # print("feature_value :", feature_value)
        # print("count :", count)
        # filter dataset sesuai dengan yang dilooping sekarang (ex : glucose rendah)
        feature_value_data = train_data[train_data[feature_name] == feature_value] 
        #print("feature_value_data :\n", pd.DataFrame(feature_value_data).head())
        
        # variabel untuk melakukan pengecekan apakah kelas sudah harus berehenti atau tidak
        assigned_to_node = False 
        cek_count = []
        cek_class = []

        for cek in class_list:
            # menghitung jumlah tiap kelas
            class_count = feature_value_data[feature_value_data[label] == cek].shape[0] 
            cek_count.append(class_count)
            cek_class.append(cek)
        
        # print("cek class :", cek_class, type(cek_class[0]))
        # print("cek count :", cek_count, type(cek_count[0]))

        for c in class_list: #looping tiap kelas
            # print("c :", c, type(c))
            # menghitung jumlah tiap kelas
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] 
            # print("Jumlah kelas",c,"adalah",class_count)

            if l_node ==True:
                if cek_count[1] > cek_count[0] :
                    #menambahkan node ke tree
                    tree[feature_value] = cek_class[1]
                    # print("tree[feature_value] :", tree[feature_value])
                    #menghilangkan baris sesuai dengan looping dari isi fitur (ex : glucose normal)
                    train_data = train_data[train_data[feature_name] != feature_value] 
                    #print("Train Data gen:\n", train_data.head())
                    assigned_to_node = True
                else : 
                    #menambahkan node ke tree
                    tree[feature_value] = cek_class[0]
                    # print("tree[feature_value] :", tree[feature_value])
                    #menghilangkan baris sesuai dengan looping dari isi fitur (ex : glucose normal)
                    train_data = train_data[train_data[feature_name] != feature_value] 
                    #print("Train Data gen:\n", train_data.head())
                    assigned_to_node = True
            elif cek_fitur != feature_name : #tambahan pengecekan
                # jumlah dari isi fitur pada tiap kelas (0/1) == jumlah kelas dari total kelas 
                if class_count == count: 
                    #menambahkan node ke tree
                    tree[feature_value] = c 
                    # print("tree[feature_value] :", tree[feature_value])
                    #menghilangkan baris sesuai dengan looping dari isi fitur (ex : glucose normal)
                    train_data = train_data[train_data[feature_name] != feature_value] 
                    #print("Train Data gen:\n", train_data.head())
                    assigned_to_node = True
            elif cek_fitur == feature_name : #tambahan pengecekan
                if cek_count[1] > cek_count[0] :
                    #menambahkan node ke tree
                    tree[feature_value] = cek_class[1]
                    # print("tree[feature_value] :", tree[feature_value])
                    #menghilangkan baris sesuai dengan looping dari isi fitur (ex : glucose normal)
                    train_data = train_data[train_data[feature_name] != feature_value] 
                    #print("Train Data gen:\n", train_data.head())
                    assigned_to_node = True
                else : 
                    #menambahkan node ke tree
                    tree[feature_value] = cek_class[0]
                    # print("tree[feature_value] :", tree[feature_value])
                    #menghilangkan baris sesuai dengan looping dari isi fitur (ex : glucose normal)
                    train_data = train_data[train_data[feature_name] != feature_value] 
                    #print("Train Data gen:\n", train_data.head())
                    assigned_to_node = True
        #pengecekan apabila jumlah dari tiap kelas masih ada atau belum bisa ditentukan berhenti atau tidak di (0/1)
        if not assigned_to_node: 
            #karena masih belum bisa ditentukan, maka diberi tanda ? (ex : rendah ?)
            tree[feature_value] = "?" 
    
    #cek_fitur = feature_name
    #print("Tree :", tree)
    # print("End GenerateSubTree\n")
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list, cek_fitur=None):
    # print("")
    # print("---> make root function <---")
    # print("prev_feature_value :", prev_feature_value)
    
    #pengecekan apabila dataset tidak berjumlah 0
    if train_data.shape[0] != 0: 
        # print("train data shape :", train_data.shape[0])
        #menemukan fitur yang memiliki information gain terbesar
        max_info_feature = find_most_informative_feature(train_data, label, class_list) 
        # print("___________________________________________________")
        # print("Max_info_feature :",  max_info_feature)
        #print("train data :", train_data)
        #mendapatkan simpil pohon dan update dataset
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list, cek_fitur) 
        # print("Cek fitur sebelumnya :", cek_fitur) 
        # print("Tree :", tree)
        #print("Train data:\n", train_data)
        next_root = None
        
        #pengecekan apabila nilai information gain sama semua
        if max_info_feature == None : 
            # print('make_tree', cek_fitur)
            #menggunakan fitur sebelumnya
            max_info_feature=cek_fitur 

        #menambahkan simpul ke dalam tree
        if prev_feature_value != None: 
            root[prev_feature_value] = dict()
            # print("if root[prev_feature_value] :", root[prev_feature_value])
            #menambahkan simpul ke max info gain
            root[prev_feature_value][max_info_feature] = tree 
            # print("root prev v :", root[prev_feature_value][max_info_feature])
            #simpul tadi digunakan untuk berlanjut ke simpul tree selanjutnya
            next_root = root[prev_feature_value][max_info_feature] 
            # print("if next root:", next_root)
        #menambahkan simpul root ke tree
        else:
            # print("add to root")
            #menambahkan cabang ke atribut yang menjadi akar
            root[max_info_feature] = tree 
            # print("Root awal :", root[max_info_feature])
            #simpul tree tadi digunakan untuk lanjut ke simpul selanjutnya
            next_root = root[max_info_feature] 
            # print("else next root:", next_root)
        
        #looping untuk cabang yang masih bisa diisi cabang baru
        for node, branch in list(next_root.items()): 
            #pengecekan apabila tree masih bisa diisi
            if branch == "?":
                # print("Max Info :", max_info_feature, "\nNode :", node, "\nBranch :", branch)
                #update dataset sesuai dengan simpul yang masih bisa diisi. 
                feature_value_data = train_data[train_data[max_info_feature] == node] 
                # print(feature_value_data)
                cek_fitur = max_info_feature
                # print("Cek fitur sekarang :", cek_fitur)
                #recursif dengan data yang sudah diupdate
                make_tree(next_root, node, feature_value_data, label, class_list, cek_fitur) 


def id3(train_data_m, label):
    # print("id3")
    train_data = train_data_m.copy() #mendapatkan salinan dataset
    tree = {} #untuk menampung tree
    list_kelas = train_data[label].unique() #mendapatkan isi kelas Outcome [0,1]
    # print("class list :", list_kelas)
    make_tree(tree, None, train_data_m, label, list_kelas) #memulai rekursif
    return tree

def predict(tree, data_cek):
    if not isinstance(tree, dict): #pengecekan jika tree sudah sampai leaf node
        return tree #return the value
    else:
        #print("tree :", tree)
        root_node = next(iter(tree)) #mendapatkan fitur pertama dari tree (ex: Glucose)
        #print("root node :", next(iter(tree)))
        isi_fitur = data_cek[root_node] #mendapatkan isi fitur dari data test sesuai dengan pengecekan atribut pertama sebelumnya
        #print("feature value :", instance[root_node])
        if isi_fitur in tree[root_node]: #mengecek apakah isi fitur sekarang ada di root sekarang
            #print("Cek :",tree[root_node][feature_value])
            return predict(tree[root_node][isi_fitur], data_cek) #lanjut ke fitur selanjutnya
        else:
            return "C2" 

def evaluate(tree, test_data_m):
    #correct_preditct = 0
    #wrong_preditct = 0
    hasil_prediksi = []
    for i in range (len(test_data_m)): #looping per data test
        #print("index :", i)
        #print("Data yang di uji sekarang :\n", pd.DataFrame(test_data_m.iloc[index]))
        #print("test_data_m.iloc[index] :", test_data_m.iloc[i])
        result = predict(tree, test_data_m.iloc[i]) #memprediksi baris yang looping sekarang
        #print("Hasil prediksi :", result)
        #print("")
        hasil_prediksi.append(result) #mendapatkan hasil prediksi dari data yang diuji
        #print("Hasil prediksinya adalah :",result)

    return hasil_prediksi

def evaluate2(tree, test_data_m):
    #correct_preditct = 0
    #wrong_preditct = 0
    hasil_prediksi = None
    for i in range (len(test_data_m)): #looping per data test
        #print("index :", i)
        #print("Data yang di uji sekarang :\n", pd.DataFrame(test_data_m.iloc[index]))
        #print("test_data_m.iloc[index] :", test_data_m.iloc[i])
        result = predict(tree, test_data_m.iloc[i]) #memprediksi baris yang looping sekarang
        #print("Hasil prediksi :", result)
        #print("")
        hasil_prediksi = result #mendapatkan hasil prediksi dari data yang diuji
        #print("Hasil prediksinya adalah :",result)

    return hasil_prediksi

def performa(y_test, y_pred):
    
    kelas = ['C1','C2','C3']
    
    confusion = confusion_matrix(y_test, y_pred)
    # print('Confusion Matrix :\n', confusion)
    
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=kelas)
    # disp.plot()
    # plt.show()
    
    a= accuracy_score(y_test, y_pred)
    a = round(a*100, 3)
    # print("akurasi =",a)
    b= precision_score(y_test, y_pred,average='weighted')
    b = round(b*100, 3)
    # print("presisi =",b)
    c= recall_score(y_test, y_pred, average='weighted')
    c = round(c*100, 3)
    # print("recall =",c)
    # d= f1_score(y_test, y_pred, average='weighted')
    # d = round(d*100, 3)
    # print("f1 score =",d)
    # print("===========================================================")

    return a, b, c

def kFold(dataset, k):
    dataC1 = dataset[dataset['Label'] == "C1"]
    dataC2 = dataset[dataset['Label'] == "C2"]
    dataC3 = dataset[dataset['Label'] == "C3"]

    kC1 = []
    kC2 = []
    kC3 = []

    fold = [] # membuat list untuk menampung data yang sudah dipecah
    for q in range(k): #looping sesuai dengan k yang telah ditentukan
        # mengambil 10% data dan memasukkannya ke dalam list 'fold'
        kC1.append(pd.DataFrame(dataC1[round((dataC1.shape[0]/k)*q):round((dataC1.shape[0]/k)*(q+1))]))
        kC2.append(pd.DataFrame(dataC2[round((dataC2.shape[0]/k)*q):round((dataC2.shape[0]/k)*(q+1))]))
        kC3.append(pd.DataFrame(dataC3[round((dataC3.shape[0]/k)*q):round((dataC3.shape[0]/k)*(q+1))]))
        #print(dataset[round((dataset.shape[0]/10)*q):round((dataset.shape[0]/10)*(q+1))])

    for j in range(k):
        fold.append(pd.concat([kC1[j],kC2[j],kC3[j]]))

    return fold

def id3biasa(data, k):
    # print(data)
    fold = kFold(data, k) # mengambil data yang sudah dibagi menjadi 10 bagian
    k = [i for i in range(k)] # membuat list berisi angka 0-9
    
    #data_bootstrap = []
    prediksi_final = []
    akurasi = []
    presisi = []
    recall = []
    
    for i in k:
        #''' SPLIT TRAIN TEST '''
        Test = fold[k[i]] #misalkan looping ke 0, maka fold[0] menjadi data test dst
        y_test = Test['Label']
        #print("Panjang data testing :", len(Test), "data")
        #print("Testing bagian ke:", i, "\n", Test)
        #print("")

        temp = k[:i]+k[i+1:] #misalkan looping ke 0, maka fold selain 0 akan menjadi data train
        Train = pd.concat([fold[y] for y in temp])
        #print("Panjang data training :", len(Train), "data")
        #print("Training bagian ke:", temp, "\n", Train)

        tree = id3(Train, 'Label') #memulai membuat tree menggunakan algoritma ID3
        #print("tree :\n", tree)

        #formatData(tree,0) #mencetak tree

        # print("Fold yang ke",i+1,"dengan data training berjumlah :", len(Train))
        # # print("Data yang akan diuji sebagai berikut :\n", pd.DataFrame(Test).head())
        # print("Panjang data test :", pd.DataFrame(Test).shape[0])
        # print("Panjang data train :", pd.DataFrame(Train).shape[0])

        y_pred = evaluate(tree, Test) #melakukan pengujian terhadap tree menggunakan data test ==> [0,1,0,1,dst]
        #print("y_temp :\n", y_temp)
        
        # print("y_pred :\n", y_pred)
        prediksi_final.append(pd.DataFrame(y_pred, columns=['P'+str(i+1)]))


        accuracy, pres, rec = performa(y_test, y_pred) #menghitung performa dari tree yg diuji sekarang
        
        akurasi.append(accuracy)
        presisi.append(pres)
        recall.append(rec)

    prediksi_final = pd.concat(prediksi_final, axis=1)
    
    return akurasi, presisi, recall, Test, Train

def aggregatting(tree, test_data, k):
    
    # print("\naggregatting\n================================================================")

    hasil_prediksi = []

    #melakukan pengujian terhadap k tree bootstrap sehingga menghasilkan k prediksi 
    for i in range(k):
        #print("Prediksi sebelum aggregat yang ke :", i+1)
        prediksi = evaluate2(tree[i], test_data)
        hasil_prediksi.append(prediksi)
    print("Hasil :",hasil_prediksi)

    # hasil_prediksi = pd.concat(hasil_prediksi, axis=1)
    # print("Hasil prediksi :",hasil_prediksi)
    count = Counter(hasil_prediksi)
    
    prediksi_final = None
    
    temp_pred = []
    temp_class = []

    for value, count in count.items():
        temp_class.append(value)
        temp_pred.append(count)
        
    print(temp_pred)
    print(temp_class)

    if len(temp_class) >= 2 :
        if temp_pred[0] >= temp_pred[1]:
            prediksi_final = temp_class[0]
        elif temp_pred[1] >= temp_pred[2]:
            prediksi_final = temp_class[1]
        else :
            prediksi_final = temp_class[2]
    elif len(temp_class) == 1 :
        prediksi_final = temp_class[0]
        #else : 
        #    prediksi_final.append(0.0)
    # print("End Aggregate")
    return prediksi_final, hasil_prediksi

def prediksi_bagging (tree, Test):
    y_pred = None
    # hasil_prediksi = []
    # for ag in range(len(tree)):
    t_y_pred, has = aggregatting(tree, Test, len(tree))
    # t_y_pred = pd.DataFrame(t_y_pred)
    # hasil_prediksi.append(has)
    y_pred = t_y_pred
    
    return y_pred