import pandas as pd #serve per leggere il file csv (sarebbe il file excel con i dati)
import numpy as np  #serve per usare gli array numpy, che sono creati col metodo array di np
import wfdb         #serve a leggere, scrivere, processare e stampare segnali e annotazioni a forma d'onda(Waveform Database)
import ast
import matplotlib.pyplot as plt    #per visualizzare un ecg come grafico
import neurokit2 as nk
import datetime
from PIL import Image   #per creare le immagini dall'array RGB
import os 

def load_raw_data(df, sampling_rate, path):
    ii= datetime.datetime.now()
    if sampling_rate == 100:
        data = np.array([wfdb.rdsamp(path+f)[0] for f in df.filename_lr],dtype=np.float64)   #itera per ogni riga sulla colonna filename_lr del dataframe
    else:
        data = np.array([wfdb.rdsamp(path+f)[0] for f in df.filename_hr],dtype=np.float64)    #itera per ogni riga sulla colonna filename_lr del dataframe
    print("intemedia ", datetime.datetime.now()-ii) 
    #trasformo l'array data in 3 array, per derivazioni I,II e V6 di ogni ecg perchè 
    #per tirare fuori le info devi prende una derivazione alla volta
    num_rows = data.shape[0] #numero righe 
    new_data = np.zeros((num_rows, 3), dtype=object) 
    for i in range(num_rows):
        #qua creo 12 array (uno per derivazione) per ogni ecg
        der=np.transpose(data[i]) 
        #qua prendo le 3 derivazioni
        new_data[i]=[der[0],der[1],der[11]]
    print("durata ", datetime.datetime.now()-ii, type(new_data), type(new_data[0])) 
    return new_data 

def array_map(f, x):
    return np.array(list(map(f, x)))   

def normalizza(l, ma, mi):
    return list(map(lambda x:(int(((x-mi)/(ma-mi))*255)),l)) 


sampling_rate=100   #sono gli hz degli ecg
# load and convert annotation data da csv a dataframe
path='C:/Users/circe/Desktop/Tirocinio/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')  #la colonna ecg_id la uso come chiave primaria diciamo
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))  #converte la colonna scp_codes da stringhe in veri e propri 
                                                                #dizionari

#questo usalo per trovare le righe con basiline_drift e far vedere che neurokit la corregge e la metti su overleaf, fatto
"""l=[]
for ecg_id,r in Y.iterrows():
    #print(r)
    if pd.notna(r['baseline_drift']):
        print(r['baseline_drift'], ecg_id)  #ecg_id: 4836 su derivazione II
        print(r)
        l.append(ecg_id)"""


#load raw signals
print("entro nella funzione")
ecg_scelto = 0#15011#4820#750   #questa è l'indice nel dataframe, si parte da 0
#I DATAFRAME PARTONO DA 0, QUINDI SE ecg_scelto è 10 parte da riga del dataframe indice 10 (primo elemento indice 0)
#print(Y.iloc[ecg_scelto:ecg_scelto+10]) 
fine=22000#2250#5000 
X = load_raw_data(Y.iloc[ecg_scelto:ecg_scelto+fine], sampling_rate, path)
#X = load_raw_data(Y, sampling_rate, path) 
print(len(X)) 
print(Y.iloc[ecg_scelto]) 


#METTI SU OVERLEAF. QUA VADO A CLASSIFICARE IN NORMALE E ANORMALE I VARI ECG. Vedendo la documentazione di ptb-xl si capisce che il campo scp_codes viene scritto dal
#cardiologo ed inoltre il numero di ecg normali combacia con quello dichiarato nella documentazione. Quindi sono normali tutti gli ecg che hanno 
#la voce NORM nel dizionario scp_codes, anormali tutti gli altri
Y['normale/anomalo'] = Y.scp_codes.apply(lambda x:'normale' if 'NORM' in x else 'anomalo') #and x['NORM']>=50 else 'abnormal')
print(Y['scp_codes'], Y['normale/anomalo'])

"""dt=dict()
c=0 
for i,r in Y[['report','scp_codes']].iterrows():
    if 'NORM' in r['scp_codes'] and r['scp_codes']['NORM']==100:
        c+=1
        if r['report'].lower() in dt:
            dt[r['report'].lower()]+=1
        else: 
            dt[r['report'].lower()]=1
print(sorted(dt.items(), key=lambda x: x[1])) 
print(c)"""



#FILTRO OGNI ECG IN X
ii= datetime.datetime.now()
print("inizio ",ii)
font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}
print("riprendo")
d_ecg_filtrati=dict()
d_battiti=dict()
minimo=20
massimo=-1

max0=0      #valori per trovare massimo e minimo di ogni derivazione 
min0=0 
max1=0
min1=0
max2=0
min2=0 
max_samples=0
n_t=-1
for ecg_id,(resto) in Y.iloc[ecg_scelto:ecg_scelto+fine].iterrows(): 
#for n_t in range(len(X)): #per ogni tracciato ecg
    n_t+=1
    #print("--------------------------------------------------------------------",ecg_scelto+n_t)
    #print("-----------------------------------------------------------ecg_id ",ecg_id) 
    #qua tolgo baseline wander e rumore (almeno in parte) 
    d_ecg_filtrati[ecg_id]=[nk.ecg_clean(X[n_t][0], sampling_rate=sampling_rate),
              nk.ecg_clean(X[n_t][1],sampling_rate=sampling_rate),
              nk.ecg_clean(X[n_t][2], sampling_rate=sampling_rate)]
    #prendo i picchi R
    if nk.ecg_invert(d_ecg_filtrati[ecg_id][2], sampling_rate=sampling_rate)[1]: #se è invertito
        p3=nk.ecg_peaks(nk.ecg_invert(d_ecg_filtrati[ecg_id][2], sampling_rate=sampling_rate)[0], sampling_rate=sampling_rate)[1]["ECG_R_Peaks"]
        print("Invertito")

    else:
        p3=nk.ecg_peaks(d_ecg_filtrati[ecg_id][2], sampling_rate=sampling_rate)[1]["ECG_R_Peaks"] 
       
    print("Picchi trovati -> ",p3)

    if len(p3)<=4: #3 
        #il 672 ne trova 4 ma sono 13 (con wfdb dovrebbe trovarli tutti 13), prova a mette 4 qua (altre due derivazini ok, , li trova facili la), 
        #12690 ha una linea piatta praticamente ma in altre due derivazioni ok.
        #19260 ha tre battiti all'inizio poi linea piatta, invece sono 11 battiti (altre due derivazioni ok, li trova facili la)
        #20006 uguale, trova 1 battito (perche ce ne sta uno nella derivazione), ma in realta sono 11(altre due derivazioni ok, li trova facili la)
        print("Meno di 3 battiti ")
        for i in range(1,-1,-1):
            if len(p3)<=4:
                #allora controllo nella derivazione 2 e poi la 1 se serve 
                print("derivazione ",i) 
                p3=nk.ecg_peaks(nk.ecg_invert(d_ecg_filtrati[ecg_id][i], sampling_rate=sampling_rate)[0], sampling_rate=sampling_rate)[1]["ECG_R_Peaks"]
                print("Trovati con neurokit -> ", p3) 
            else:
                break 

    p3=np.unique(p3)    #tolgo valori uguali se ci sono

    if any(x<0 for x in p3):    #se ci sono indici negativi li toglie (con neurokit non lo fa)
        p3=np.delete(p3, np.where(p3<0))
        p3=np.delete(p3,np.where(p3>999)) 
        print("Valori negativi tolti ",p3)
        input("bo")

    p=p3 
    e=1
    f=False
    while e <len(p3):  # con neurokit non lo fa
        if p3[e]-p3[e-1]<=15:
            p3=np.delete(p3,e)
            f=True
            continue 
        e+=1
    if f:
        print("Valori vicini tolti ",p3) 
        if p!=p3:
            input("diverso") 


    print(" picchi trovati -> ", p3) 
    minimo=min(minimo,len(p3))
    massimo=max(massimo, len(p3))
    print(minimo, massimo) 

    #segmentazione 
    battiti0 = []
    battiti1 = []
    battiti2 = []
    d_battito_piu_grande_ecg=-1
    bspos=[]

    #segmentazione primo battito
    diff = p3[1] - p3[0]  
    i = max((p3[0] - diff//2),0) #posizione inizio del battito, perchè se è negativo allora parte dall'inizio dell'ecg
    f = (p3[0] + diff//2) - 1 #posizione fine del battito
    battiti0.append( d_ecg_filtrati[ecg_id][0][i:f] )
    battiti1.append( d_ecg_filtrati[ecg_id][1][i:f] )
    battiti2.append( d_ecg_filtrati[ecg_id][2][i:f] )
    bspos.append((i,f))
    max_samples=max(max_samples, (f-i)) 
    #aggiunto per trovare il battito piu grande nel singolo ecg (per creare immagini larghe uguali per ogni ecg)
    d_battito_piu_grande_ecg=max(d_battito_piu_grande_ecg, f-i)

    #segmentazione battiti in mezzo
    count = 1 
    for i in (p3[1:-1]):
        diff1 = abs(p3[count - 1] - i)
        diff2 = abs(p3[count + 1]- i) 
        i = p3[count - 1] + diff1//2 #posizione inizio del battito
        f = (p3[count + 1] - diff2//2)-1 #posizione fine del battito
        battiti0.append( d_ecg_filtrati[ecg_id][0][i:f] )
        battiti1.append( d_ecg_filtrati[ecg_id][1][i:f] )
        battiti2.append( d_ecg_filtrati[ecg_id][2][i:f] )
        count += 1
        bspos.append((i,f)) 
        max_samples=max(max_samples, (f-i)) 
        #aggiunto per trovare il battito piu grande nel singolo ecg (per creare immagini larghe uguali per ogni ecg)
        d_battito_piu_grande_ecg=max(d_battito_piu_grande_ecg, f-i)

    #segmentazione primo battito
    diff = p3[-1] - p3[-2]   
    i = p3[-1] - diff//2     #posizione inizio del battito
    f = min((p3[-1] + diff//2) - 1, 999) #posizione fine del battito, perchè se è maggiore di 999 allora finisce fuori dall'ecg
    battiti0.append( d_ecg_filtrati[ecg_id][0][i:f] )
    battiti1.append( d_ecg_filtrati[ecg_id][1][i:f] )
    battiti2.append( d_ecg_filtrati[ecg_id][2][i:f] )
    bspos.append((i,f))
    max_samples=max(max_samples, (f-i))
    #aggiunto per trovare il battito piu grande nel singolo ecg (per creare immagini larghe uguali per ogni ecg)
    d_battito_piu_grande_ecg=max(d_battito_piu_grande_ecg, f-i)

    d_battiti[ecg_id]=[battiti0,battiti1,battiti2, d_battito_piu_grande_ecg]  

    #trovo i valori massimi e minimi per ogni derivazione, utile per normalizzazione
    #tra 0 e 255
    max0=max(max0, max(map(lambda x:max(x), battiti0 )))
    min0=min(min0, min(map(lambda x:min(x), battiti0 )))
    max1=max(max1, max(map(lambda x:max(x), battiti1 )))
    min1=min(min1, min(map(lambda x:min(x), battiti1 )))
    max2=max(max2, max(map(lambda x:max(x), battiti2 ))) 
    min2=min(min2, min(map(lambda x:min(x), battiti2 )))

print("durata elaborazione ecg :", datetime.datetime.now()-ii)
    
print("derivazione 0 finale -> ", max0, min0)
print("derivazione 1 finale -> ", max1, min1)
print("derivazione 2 finale -> ", max2, min2)
print("battito piu grande ha-> ",max_samples, " punti") 


#qua organizzo la cartella dove mettere le immagini che saranno prodotte, nello stesso modo del dataset originario
pi='C:/Users/circe/Desktop/Tirocinio/'
d_s='Immagini generate larghezza battito ecg/'
os.makedirs(pi+d_s, exist_ok=True) 
for i in range(22):
    os.makedirs(pi+d_s+'0'*(5-len(str(i*1000)))+str((i*1000)), exist_ok=True) 

i=datetime.datetime.now()
for ecg_id, (der1, der2, der3, bg) in d_battiti.items():   # Cicla direttamente su ogni coppia di derivazioni 
    # Normalizza i battiti di ogni derivazione tra 0 e 255
    b0n = [normalizza(val, max0, min0) for val in der1] 
    b1n = [normalizza(val, max1, min1) for val in der2]
    b2n = [normalizza(val, max2, min2) for val in der3]
    
    # Popolare la matrice con i valori dei canali normalizzati
    nrighe = len(b0n)   # Numero di righe=numero di battiti
    #ncolonne = max_samples  # Numero colonne=massimo numero di samples di tutte le derivazioni
    ncolonne=bg 
    immagine = np.zeros((nrighe, ncolonne, 3), dtype=np.uint8) 
    for i, (r, g, b) in enumerate(zip(b0n, b1n, b2n)):
        for j, color in enumerate(zip(r, g, b)):
            immagine[i, j] = list(color)  # Usa il valore normalizzato per tutti i canali RGB

    img = Image.fromarray(immagine)  
    #img.save("Immagini generate/" + f"immagine_ecg_{ecg}.png") 
    #img.save("Immagini generate larghezza battito ecg/" + f"{'0'*(5-len(str(ecg_id)))+str(ecg_id)}.png")  
    img.save(pi+d_s+ f"{'0'*(5-len(str((ecg_id//1000)*1000)))+str((ecg_id//1000)*1000)+'/'+'0'*(5-len(str((ecg_id//1000)*1000)))+str(ecg_id)}.png" )
    #img.save("Monnezze/"+f"{'0'*(5-len(str(ecg_id)))+str(ecg_id)}.png")
    d_battiti[ecg_id] = [b0n, b1n, b2n] 
     
print("FINE")
    
""" 
#METTI POI SU OVERLEAF. questo usato solo per spostare le immagini nella cartella nella cartella corrispondente, per avere organizzazione delle immagini come il dataset
#originale
for f in os.listdir(p):
    if ".png" in f: 
        cc=(int(f[:-4])//1000) * 1000
        cc='0'*len(5-len(cc)) + str(cc) 
        os.rename(p+"/"+f, p+"/"+cc+"/"+f)
"""
