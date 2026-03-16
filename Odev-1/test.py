import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CIFAR-10 veri batch dosyalarını okumak için fonksiyon
# CIFAR veri seti pickle formatında saklandığı için
# pickle.load kullanarak dosya içeriğini okuyoruz
# ---------------------------------------------------------
def load_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    
    # görüntü piksel değerleri
    data = data_dict[b'data']
    
    # görüntülerin sınıf etiketleri
    labels = data_dict[b'labels']
    
    return data, labels


# ---------------------------------------------------------
# CIFAR-10 veri setinin bulunduğu klasör
# ---------------------------------------------------------
data_dir = "cifar-10-batches-py"


# ---------------------------------------------------------
# Eğitim verilerini tutmak için listeler oluşturuluyor
# CIFAR-10 eğitim verisi 5 ayrı batch dosyasından oluşur
# ---------------------------------------------------------
X_train_list = []
y_train_list = []


# ---------------------------------------------------------
# data_batch_1 ... data_batch_5 dosyalarını tek tek okuyup
# eğitim veri setini oluşturuyoruz
# ---------------------------------------------------------
for i in range(1,6):
    
    # ilgili batch dosyasını yükle
    data, labels = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
    
    # görüntü verilerini listeye ekle
    X_train_list.append(data)
    
    # etiketleri listeye ekle
    y_train_list.extend(labels)


# ---------------------------------------------------------
# Tüm batchleri tek bir eğitim veri setinde birleştiriyoruz
# ---------------------------------------------------------
X_train = np.concatenate(X_train_list)

# etiket listesini numpy array'e çeviriyoruz
y_train = np.array(y_train_list)


# ---------------------------------------------------------
# Test veri setini yükleme
# ---------------------------------------------------------
X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))


# ---------------------------------------------------------
# Veri seti boyutlarını ekrana yazdırma
# ---------------------------------------------------------
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


# ---------------------------------------------------------
# Piksel değerlerini normalize etme
# CIFAR görüntülerinin piksel değerleri 0-255 arasındadır
# Bunları 0-1 aralığına ölçekleyerek hesaplamaları
# daha stabil hale getiriyoruz
# ---------------------------------------------------------
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


# ---------------------------------------------------------
# CIFAR-10 veri setindeki sınıf isimleri
# ---------------------------------------------------------
label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ---------------------------------------------------------
# Kullanıcıdan mesafe metriği seçimi
# L1 : Manhattan Distance
# L2 : Euclidean Distance
# ---------------------------------------------------------
while True:
    metric = input("Mesafe türünü seçin (L1 / L2): ").upper()
    
    if metric in ["L1", "L2"]:
        break
    else:
        print("Lütfen sadece L1 veya L2 girin.")


# ---------------------------------------------------------
# Kullanıcıdan k değeri alınır
# k değeri, k-NN algoritmasında kaç komşunun dikkate
# alınacağını belirler
# ---------------------------------------------------------
k = int(input("k değerini girin: "))


# ---------------------------------------------------------
# Test veri setinden rastgele bir görüntü seçiyoruz
# ---------------------------------------------------------
test_index = np.random.randint(0, len(X_test))

# seçilen test görüntüsü
test_image = X_test[test_index]


# ---------------------------------------------------------
# Tüm eğitim görüntüleri ile test görüntüsü arasındaki
# mesafeleri hesaplamak için boş liste
# ---------------------------------------------------------
distances = []


# ---------------------------------------------------------
# Eğitim veri setindeki her görüntü ile test görüntüsü
# arasındaki mesafeyi hesaplıyoruz
# ---------------------------------------------------------
for i in range(len(X_train)):

    train_image = X_train[i]

    # Manhattan Distance (L1)
    if metric == "L1":
        dist = np.sum(np.abs(test_image - train_image))

    # Euclidean Distance (L2)
    elif metric == "L2":
        dist = np.sqrt(np.sum((test_image - train_image) ** 2))

    distances.append(dist)


# ---------------------------------------------------------
# Mesafeleri numpy array formatına çeviriyoruz
# ---------------------------------------------------------
distances = np.array(distances)


# ---------------------------------------------------------
# En küçük k mesafeye sahip eğitim örneklerini buluyoruz
# ---------------------------------------------------------
nearest_indices = np.argsort(distances)[:k]


# ---------------------------------------------------------
# Bu k komşunun etiketlerini alıyoruz
# ---------------------------------------------------------
nearest_labels = [y_train[i] for i in nearest_indices]


# ---------------------------------------------------------
# Majority voting (çoğunluk oyu)
# En fazla tekrar eden sınıf tahmin edilir
# ---------------------------------------------------------
predicted_label = np.bincount(nearest_labels).argmax()


# ---------------------------------------------------------
# Tahmin edilen ve gerçek sınıfı ekrana yazdırıyoruz
# ---------------------------------------------------------
print("\nTahmin edilen sınıf:", label_names[predicted_label])
print("Gerçek sınıf:", label_names[y_test[test_index]])


# ---------------------------------------------------------
# Görüntüyü ekranda göstermek için reshape işlemi
# CIFAR görüntüleri (3,32,32) formatındadır
# bunu (32,32,3) formatına çeviriyoruz
# ---------------------------------------------------------
image = X_test[test_index].reshape(3, 32, 32)

# RGB formatına dönüştürme
image = image.transpose(1, 2, 0)


# ---------------------------------------------------------
# Görüntüyü matplotlib ile gösterme
# ---------------------------------------------------------
plt.figure(figsize=(4,4))
plt.imshow(image)

# Tahmin edilen sınıfı başlık olarak yazdırma
plt.title("Prediction: " + label_names[predicted_label])

plt.axis("off")
plt.show()