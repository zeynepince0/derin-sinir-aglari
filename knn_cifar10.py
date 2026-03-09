import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    return data, labels


data_dir = "cifar-10-batches-py"

X_train_list = []
y_train_list = []

for i in range(1,6):
    data, labels = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
    X_train_list.append(data)
    y_train_list.extend(labels)

X_train = np.concatenate(X_train_list)
y_train = np.array(y_train_list)


X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

while True:
    metric = input("Mesafe türünü seçin (L1 / L2): ").upper()
    if metric in ["L1", "L2"]:
        break
    else:
        print("Lütfen sadece L1 veya L2 girin.")

k = int(input("k değerini girin: "))


test_index = np.random.randint(0, len(X_test))
test_image = X_test[test_index]


distances = []

for i in range(len(X_train)):

    train_image = X_train[i]

    if metric == "L1":
        dist = np.sum(np.abs(test_image - train_image))

    elif metric == "L2":
        dist = np.sqrt(np.sum((test_image - train_image) ** 2))

    distances.append(dist)


distances = np.array(distances)

nearest_indices = np.argsort(distances)[:k]

nearest_labels = [y_train[i] for i in nearest_indices]

predicted_label = np.bincount(nearest_labels).argmax()


print("\nTahmin edilen sınıf:", label_names[predicted_label])
print("Gerçek sınıf:", label_names[y_test[test_index]])


image = X_test[test_index].reshape(3, 32, 32)
image = image.transpose(1, 2, 0)

plt.figure(figsize=(4,4))
plt.imshow(image)
plt.title("Prediction: " + label_names[predicted_label])
plt.axis("off")
plt.show()