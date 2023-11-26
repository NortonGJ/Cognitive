import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from tqdm import tqdm, trange
from random import shuffle as rand_shuffle
import gc, os

datadir = './SDRPlay-RPS1/data_50'
filename = '15_50_60000'
savedir = 'SDR_test'

if not os.path.exists(f'{datadir}/{savedir}'):
    os.mkdir(f'{datadir}/{savedir}')

def conv(value, From1 = -120, From2 = -30, To1 = -1, To2 = 0):
    return (value - From1) / (From2 - From1) * (To2 - To1) + To1

def randShuffle(seed: int, array: np.ndarray):
    np.random.seed(seed)
    np.random.shuffle(array)
    

temp = np.load(f'{datadir}/{filename}.npy')

PnoiseMin = -80 # Мин мощность помехи
PnoiseMax = -30 # Макс мощность помехи
Pheat = -120    # Мощность тепл шума
# PdBmNoise = PnoiseMax # Мощность доп помехи
La = temp.shape[1]   # Число каналов
T = temp.shape[0] # Время генерации
Tsize = 60
EPS = 10
Pnoise = 5/La
Tnoise = 5
size = (T * (T-Tsize), La)
preNoise = uniform(0.0, 1.0, temp.shape)
newData = np.copy(temp)

# Наложение помехи на исходный непрерывный набор данных
for channel in trange(La):
    time_ = 0
    while time_ < T:
        is_noise = preNoise[time_, channel] < Pnoise
        if is_noise:
            t_start = time_
            t_end = time_ + Tnoise if (time_+Tnoise) < T else T
            newData[t_start:t_end, channel] = PnoiseMax * np.ones(t_end-t_start)
            time_ = t_end
        time_ += 1

label_s = -1
label_l = -1
label_i = -1

predRes = []
predLabels_s = []
predLabels_l = []
predLabels_i = []

#  Разбиение на вектора для обучения с сохранением labels: _s, _l, _i
for time_ in trange(T - Tsize):
    # В зависимости от того, учитывается ли пред предсказание
    # label_s = -1
    # label_l = -1
    # label_i = -1
    p0_split = np.zeros((Tsize, La), dtype=np.uint8)
    E_split = newData[time_:time_+Tsize]
    Elast = E_split[-1]
    ENorm = (-1.) * conv(E_split)
    ENormlast =  ENorm[:, -1]
    
    p0_split[E_split < PnoiseMin] = 1
    PT_split = np.mean(p0_split, axis=0)
    # Краткосрочный
    Emin = np.min(Elast)
    E_check = (np.abs(Elast - Emin) <= EPS)
    indexes = np.nonzero(E_check)[0]
    if label_s not in indexes:
        label_s = np.random.choice(indexes)
    Es = Elast[label_s]; Ps = PT_split[label_s]
    # Долгосрочный
    adaptiveLvl = 0.9    
    adaptiveLvl2 = 1.0
    while True:
        PTmax = np.max(PT_split)
        PT_check = (PT_split >= PTmax * adaptiveLvl)
        if len(PT_check) == 0:
            adaptiveLvl -= 0.1
            continue
        indexes = np.nonzero(PT_check)[0]
        indexes = indexes[Elast[indexes] < PnoiseMin * adaptiveLvl2]
        if len(indexes) == 0:
            adaptiveLvl2 -= 0.05
            continue
        if label_l not in indexes and len(indexes) != 0:
            label_l = np.random.choice(indexes)
        break
    El = Elast[label_l]; Pl = PT_split[label_l]
    # Интегральный
    label_i = label_s if np.uint8(Es/El < Ps/Pl) else label_l 
    # Запись неперемешанных векторов и разметки
    predRes.append(ENorm.reshape(-1))
    
    label_array = np.zeros(La)
    label_array[label_s] = 1
    predLabels_s.append(label_array)
    
    label_array = np.zeros(La)
    label_array[label_l] = 1
    predLabels_l.append(label_array)
    
    label_array = np.zeros(La)
    label_array[label_l] = 1
    predLabels_l.append(label_array)

print('save predShuffle')    
with open(f"{datadir}/{savedir}/{filename}_vec_noShuffle.npy", 'wb') as f:
    np.save(f, predRes)

print('shuffle')   
# Шаффл векторов на обучение и разметки
seed = np.random.randint(0, 10000)
np.random.seed(seed)
np.random.shuffle(predRes)

np.random.seed(seed)
np.random.shuffle(predLabels_s)

np.random.seed(seed)
np.random.shuffle(predLabels_l)

np.random.seed(seed)
np.random.shuffle(predLabels_i)

print('save')                  
with open(f"{datadir}/{savedir}/{filename}_vec.npy", 'wb') as f:
    np.save(f, predRes)
    
with open(f"{datadir}/{savedir}/{filename}_labels_s.npy", 'wb') as f:
    np.save(f, predLabels_s)
with open(f"{datadir}/{savedir}/{filename}_labels_l.npy", 'wb') as f:
    np.save(f, predLabels_l)
with open(f"{datadir}/{savedir}/{filename}_labels_i.npy", 'wb') as f:
    np.save(f, predLabels_i)