from PIL import Image as IM
import PIL.ImageOps
import numpy as np
import math, sys

def imsaveFromArray(array, name: str, invFlag = True):
    img = IM.fromarray(array)
    img = img.convert("L")
    print("saved")
    if (invFlag == True):
        img = PIL.ImageOps.invert(img)
        img.save(name+'.jpg')
    else:
        img.save(name+"_noInv"+'.jpg')

def imshowFromArray(array, invFlag = True):

    img = IM.fromarray(array)
    img = img.convert("L")
    if (invFlag == True):
        img = PIL.ImageOps.invert(img)
    img.show()

def imsaveColorFromArray(colorM, name):

    img = IM.fromarray(colorM, 'RGB')
    img.save(name+'.jpg')
    print("saved")

def imshowColorFromArray(colorM):
   
    img = IM.fromarray(colorM, 'RGB')
    img.show()

def conv(value: np.ndarray, From1=-120, From2=-30, To1=-1, To2=0):
    return (value - From1) / (From2 - From1) * (To2 - To1) + To1

def showPallete(Pallete: np.ndarray, save = False, show = True):
    num = np.shape(Pallete)[0]
    base_height = 60
    height, width = base_height * num, 320
    border = 3
    img_array = np.zeros((height, width, 3), dtype = np.uint8)
    for i in range(num):
        color = Pallete[i]
        img_array[i*base_height:(i + 1)*base_height,:] = color
    img = np.ones((height + 2 * border, width + 2 * border, 3), dtype = np.uint8) * 255
    img[border:border+height, border:border+width] = img_array
    if save:
        imsaveColorFromArray(img, 'base_pallete_3')
    if show:
        imshowColorFromArray(img)

def upscale(matrix: np.ndarray, x_multi: int, y_multi:int):
    base_x, base_y = np.shape(matrix)
    new_x = base_x * x_multi
    new_y = base_y * y_multi
    
    new_matrix = np.zeros((new_x, new_y))
    for i in range(0, new_x, x_multi):
        for j in range(0, new_y, y_multi):
            new_matrix[i, j] = matrix[i//x_multi, j//y_multi]
    for i in range(base_x):
        new_matrix[i * x_multi: i * x_multi + x_multi] = new_matrix[i * x_multi]
    new_matrix = np.transpose(new_matrix)
    for j in range(base_y):
        new_matrix[j * y_multi: j * y_multi + y_multi] = new_matrix[j * y_multi]
    new_matrix = np.transpose(new_matrix)
    return new_matrix

def downscale(PdT: np.ndarray, factor: int):
    baseFreqLen = np.len(PdT)
    newFreqLen = math.ceil(baseFreqLen / factor)
    PdTscaled = np.zeros(newFreqLen, dtype = PdT.dtype)
    for i in range(newFreqLen - 1):
        PdTscaled[i] = np.mean(PdT[factor * i: factor * (i + 1)])
    PdTscaled[-1] = np.mean(PdT[factor * (i + 1): -1])
    return PdTscaled

def recolor(matrix: np.ndarray, pallete, lvls: np.ndarray):
    if (len(lvls) + 1 != np.shape(pallete)[0]):
        print('pallete/lvls error')
        sys.exit(-1)
    base_size = []
    eq = 1
    for elem in matrix.shape:
        base_size.append(elem)
        eq *= elem
    base_size.append(3)
    img = np.zeros((eq, 3), dtype = np.uint8)
    for ind, elem in enumerate(matrix.reshape(-1)):
        if elem <= lvls[0]:
            img[ind] = pallete[0]
        elif elem >= lvls[-1]:
            img[ind] = pallete[-1]
        for ii in range(1, len(lvls)):
            if (lvls[ii - 1] <= elem < lvls[ii]):
                img[ind] = pallete[ii]
                break
    return img.reshape(base_size)

def addGrid(img, x_step: int, y_step: int, color = [255,255,255]):
    for ii in range(x_step, img.shape[0], x_step):
        img[ii] = color
    for jj in range(y_step, img.shape[1], y_step):
        img[:,jj] = color
    return img    

def getPredict(data: np.ndarray, method: int, criterion: int, prev_predict = -1):
    T = 60
    E_Noise = -80
    INDshort, INDlong = prev_predict, prev_predict
    EPS = 5
    
    if (data.shape[0]) != T:
        data = data.transpose()
        
    E0 = data
    Elast = E0[-1]
    p = np.zeros_like(E0, dtype = np.uint8)
    p[E0 < E_Noise] = 1
    PT = np.sum(p, axis = 0) / T
    if method == 1:
        # Краткосрочный
        Emin = np.min(Elast)
        E_check = (np.abs(Elast - Emin) <= EPS)
        indexes = np.nonzero(E_check)[0]
        if INDshort not in indexes:
            INDshort = np.random.choice(indexes)
        Es = Elast[INDshort]; Ps = PT[INDshort]
        # Долгосрочный
        adaptiveLvl = 0.9    
        adaptiveLvl2 = 1.0
        while True:
            PTmax = np.max(PT)
            PT_check = (PT >= PTmax * adaptiveLvl)
            if len(PT_check) == 0:
                adaptiveLvl -= 0.1
                continue
            indexes = np.nonzero(PT_check)[0]
            indexes = indexes[Elast[indexes] < E_Noise * adaptiveLvl2]
            if len(indexes) == 0:
                adaptiveLvl2 -= 0.05
                continue
            if INDlong not in indexes and len(indexes) != 0:
                INDlong = np.random.choice(indexes)
            break
        El = Elast[INDlong]; Pl = PT[INDlong]
        # Интегральный
        INDintegral = INDlong if np.uint8(Es/El < Ps/Pl) else INDlong 
        pred = [INDshort, INDlong, INDintegral]
        Predict = pred[criterion]
    return Predict

def mergeData(dataDir, filename, iStart, iEnd):
    temp = np.load(f'{dataDir}/{iStart}.npy')
    for index in range(iStart+1, iEnd, 1):
        temp = np.vstack((temp, np.load(f'{dataDir}/{index}.npy')))
    with open (f"{dataDir}/{filename}.npy", "wb") as File:
        np.save(File, temp)