from random import shuffle
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense, Softmax, Flatten, Dropout, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Softmax, Flatten, Dropout, Conv2D, MaxPool2D
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image 
import os 
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score, auc, roc_curve, classification_report
import seaborn as sns 
"""
#In TensorFlow, le funzioni compile, fit, evaluate e predict sono utilizzate per addestrare, valutare e utilizzare reti neurali per compiti di machine learning e deep learning. Ecco una breve spiegazione di ognuna di queste funzioni:
#compile: La funzione compile viene utilizzata per configurare il processo di addestramento di un modello. Accetta diversi argomenti importanti, tra cui:
#optimizer: Specifica l'ottimizzatore da utilizzare durante l'addestramento (ad esempio, "adam", "sgd", ecc.).
#loss: Specifica la funzione di perdita (loss function) che misura l'errore del modello durante l'addestramento. Ad esempio, per problemi di classificazione binaria, si usa spesso "binary_crossentropy", mentre per problemi di classificazione multiclasse, si usa "categorical_crossentropy".
#metrics: Specifica le metriche da calcolare durante l'addestramento (ad esempio, "accuracy" per la classificazione).
#Ecco un esempio di utilizzo della funzione compile:
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fit: La funzione fit viene utilizzata per addestrare effettivamente il modello sui dati di addestramento. Accetta gli input (dati di addestramento) e le etichette (ground truth) come argomenti principali, oltre a molte altre opzioni, tra cui il numero di epoche di addestramento e il batch size. Durante il processo di addestramento, il modello viene aggiornato iterativamente per minimizzare la funzione di perdita specificata durante la compilazione.
#Ecco un esempio di utilizzo della funzione fit:
#model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#evaluate: La funzione evaluate viene utilizzata per valutare le prestazioni del modello su un set di dati di test separato. Accetta gli input di test e le etichette di test come argomenti e restituisce le metriche specificate durante la compilazione. Questo � utile per valutare quanto bene il modello si comporta su dati che non ha mai visto durante l'addestramento.
#Ecco un esempio di utilizzo della funzione evaluate:
#loss, accuracy = model.evaluate(X_test, y_test)

#predict: La funzione predict viene utilizzata per ottenere le previsioni del modello su nuovi dati (senza etichette) una volta che il modello � stato addestrato. Accetta gli input su cui effettuare le previsioni e restituisce le previsioni stesse. Questo � utile per l'utilizzo del modello per scopi di inferenza dopo l'addestramento.
#Ecco un esempio di utilizzo della funzione predict:
#predictions = model.predict(X_new_data)

#In sintesi, compile configura il modello, fit addestra il modello, evaluate valuta il modello su dati di test e predict utilizza il modello per fare previsioni su nuovi dati. Queste funzioni sono fondamentali quando si lavora con reti neurali in TensorFlow per una vasta gamma di problemi di machine learning e deep learning.
 
"""


"""def load_image(file_name):
    raw = tf.io.read_file(file_name)
    tensor = tf.io.decode_image(raw)
    #print(tensor[:,:,:-1])
    tensor = tf.image.rgb_to_grayscale(tensor) 

    return tensor"""

"""path = 'C:/Users/circe/Desktop/Tirocinio/Immagini classificate/'
l_i=np.array([]) 
for c in os.listdir(path): 
    for i in os.listdir(path+c)[:2]: 
        #img=Image.open(path+c+i) 
        print(path+c+'/'+i) 
        np.append(l_i, load_image(path+c+'/'+i)) 

        #print(load_image(path+c+'/'+i))
        #plt.imshow(load_image(path+c+'/'+i), cmap='gray')
        #plt.show()
    print(l_i) 
    break """

#m = tf.keras.models.load_model('C:/Users/circe/Desktop/Tirocinio/modello80.keras')
#print(m)

# Definisci il percorso alla cartella principale dei dati
original_dir = 'C:/Users/circe/Desktop/Tirocinio/Immagini classificate - 80 10 10/Train validazione/'
test_dir = 'C:/Users/circe/Desktop/Tirocinio/Immagini classificate - 80 10 10/Test/'
n_ri = 110#100#128#224 
n_col = 110#100#128#224
# Definisci le dimensioni target delle immagini
target_size = (n_ri, n_col)  # Cambia queste dimensioni in base alle tue esigenze

# Crea un generatore di immagini per il set di addestramento
datagen = ImageDataGenerator(
    validation_split=0.112, # Imposta la suddivisione tra addestramento e convalida 
    rescale=1.0/255.0       # Ridimensiona i valori dei pixel tra 0 e 1
    )

# Carica e prepara i dati di addestramento
train_generator = datagen.flow_from_directory(
    original_dir,
    target_size=target_size,
    batch_size=32,  
    class_mode="binary",  # Poich� ho due classi: "Normale" e "Anomalo"
    subset="training",  # Specifica che questo � il set di addestramento
    )

# Carica e prepara i dati di convalida
validation_generator = datagen.flow_from_directory(
    original_dir,
    target_size=target_size,
    batch_size=32,  
    class_mode="binary",  # Poich� ho due classi: "Normale" e "Anomalo"
    subset="validation",  # Specifica che questo � il set di validation 
    )
#ANOMALO E' 0 E NORMLAE E' 1 (va in ordine alfabetico)
test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
    )

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size =  target_size,
    batch_size = 32, 
    class_mode = 'binary',
    shuffle = False,
    save_to_dir = 'C:/Users/circe/Desktop/nonso'
    )
#prova a separare te l'insieme di validazione (questo sopra) con un ciclo for in validazione e test (tanto il fit accetta pure gli array numpy (feature ,etichetta ))

"""
model = Sequential([
    Conv2D(8, 3, input_shape = (n_ri,n_col,3), padding='same'),
    MaxPool2D(),
    Conv2D(4, 3, padding='same'),
    MaxPool2D(),
    Flatten(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
 """

for f, t in train_generator: 
    print(type(f),type(t))              # stampa <class 'numpy.ndarray'> <class 'numpy.ndarray'> 
    print(np.shape(f), np.shape(t))     # (32, 224, 224, 3) (32,) , 32 � la dimensione del batch, 224 e 224 sono larghezza e altezza e 3 perch� rgb
    #print("f -> ", f," t -> ",f)
    break 

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(n_ri,n_col,3), padding='same'),
    MaxPool2D(),
    Conv2D(64, 3, activation='relu', padding='same'), 
    MaxPool2D(), 
    Conv2D(128, 3, activation='relu', padding='same'), 
    MaxPool2D(), 
    Flatten(),
    Dense(128, activation='relu'), 
    Dropout(0.5),
    Dense(1, activation='sigmoid')
    ]) 
print(model.summary() ) 
model.compile( 
    loss='binary_crossentropy',
    metrics=['accuracy'] 
              )

#quello che printa durante il fit sono i batch che sta analizzando. Dato che ci sono 545 batch ognuno da 32 immagini si hanno infatti 17 440 immagini di train

history = model.fit(train_generator, 
                    epochs=20, 
                    validation_data=validation_generator,
                    callbacks = [early_stopping]
                    ) 
 

y_pred = model.predict(test_generator)
y_true = test_generator.labels 
y_pred_class = (y_pred > 0.5).astype(int) 

cm = confusion_matrix(y_true, y_pred_class)
print(cm) 
#array([[1030,  198],
#       [ 202,  749]], dtype=int64)
"""
TP = 749
FN = 202
FP = 198
TN = 1030
Il Tasso di Veri Positivi (TPR), noto anche come sensibilit� o recall, � calcolato come TP / (TP + FN). Quindi nel tuo caso, TPR = 749 / (749 + 202) = 0.79.

Il Tasso di Falsi Positivi (FPR), noto anche come fall-out, � calcolato come FP / (FP + TN). Quindi nel tuo caso, FPR = 198 / (198 + 1030) = 0.16.
"""

# Crea un DataFrame per una migliore visualizzazione
cm_df = pd.DataFrame(cm)

# Visualizza la matrice di confusione come heatmap
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Reds')
plt.title('Matrice di Confusione')
plt.xlabel('Valori Predetti')
plt.ylabel('Valori Veri')
plt.show()


print(roc_auc_score(y_true, y_pred_class)) 
#0.8131771116979555

# Calcola la curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred) 
# Crea un grafico
plt.figure()
# Disegna la curva ROC
plt.plot(fpr, tpr, label='Modello (area = %0.2f)' % roc_auc_score(y_true, y_pred_class))
# Disegna la linea diagonale
plt.plot([0, 1], [0, 1],'r--')
# Imposta i limiti degli assi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# Aggiungi titolo e etichette
plt.xlabel('Tasso di falsi positivi')
plt.ylabel('Tasso di veri positivi')
plt.title('Curva ROC')
plt.legend(loc="lower right")
# Mostra il grafico
plt.show()



print(classification_report(y_true, y_pred_class)) 
