import sqlite3
import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute("SELECT name, embedding FROM faces")

list_names = []
list_vect = []
for name, embedding_bytes in cursor.fetchall():
    stored_encoding = np.frombuffer(embedding_bytes, dtype=np.float64)
    list_names.append(name)
    list_vect.append(stored_encoding)
    print(name, stored_encoding)
    #match = face_recognition.compare_faces([stored_encoding], face_encoding)
conn.close()
matr = np.zeros([int(len(list_vect)),int(len(list_vect))])
for i in range(len(list_vect)):
    a=list_vect[i]
    for j in range(len(list_vect)):
        b=list_vect[j]
        cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        matr[i,j] = cosine_similarity



# Пример данных
#matr = np.random.rand(4, 4)
col_labels = list_names
row_labels = list_names

# Создаем изображение матрицы
plt.imshow(matr, cmap='viridis')

# Настраиваем подписи осей
ax = plt.gca()
ax.set_xticks(np.arange(len(list_names)))          # Позиции тиков для столбцов
ax.set_xticklabels(col_labels)       # Подписи столбцов
ax.set_yticks(np.arange(len(list_names)))          # Позиции тиков для строк
ax.set_yticklabels(row_labels)       # Подписи строк

# Дополнительные настройки (опционально)
plt.colorbar(label='Значения')       # Цветовая шкала
ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)  # Тики снизу

plt.show()


#plt.imshow(matr)

#plt.matshow(matr, cmap=plt.cm.hot)
plt.show()