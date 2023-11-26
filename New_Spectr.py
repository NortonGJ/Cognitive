# -*- coding: utf-8 -*-
"""
dynamic graph
"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FixedFormatter

import numpy as np
import time, sys

from .func import showPallete, upscale, recolor

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.setLayout(vertical_layout)

if __name__ == '__main__':
    from sys import argv, exit as sys_exit
    app = QtWidgets.QApplication(argv)
    app.setQuitOnLastWindowClosed(True)
    application = Window()
    application.show()

    T = 60; La = 50
    fs = 10e6              # Radio sample Rate
    freq = 15e6            # LO tuning frequency in Hz
    fMEG = freq//1e6
    fsMEG = fs//1e6
    COLOR_PALLETE = [[0, 0, 0],[23, 23, 23],[51,51,51],[71,71,71],[93,93,93],
                    [37,18,61],[69,20,55],[108,24,50],[139,25,45],[195,30,36],[221,32,32]]
    #Меньше -120, -120 -110, -110 -100, -100 -90, -90 -80,
    #       -80 -70, -70 -60, -60 -50, -50 -40, -40 -30, больше -30
    COLOR_PRED = [0,255,0] # Зеленый
    cl = np.shape(COLOR_PALLETE)[0]
    lvl = np.linspace(start = -120, stop = -30, num = cl - 1)
    showPallete(COLOR_PALLETE, show = False)

    E_NoiseMAX = np.array((-80, -80, -80))
    E_Noise = -80
    datadir = './SDRPlay-RPS1/data_50/SDR_test'
    filename = '15_50_60000_noise'
    x = np.load(f'{datadir}/{filename}.npy')
    # x = np.load("test4.npy")
    # x = x.transpose()
    # x = 10*(np.log10(x) + 3)
    E = x[0:T]
    base_x, base_y = E.shape

    channel_width = 5; time_height = 5
    height = base_x * time_height
    width = base_y * channel_width

    Escale = upscale(E, time_height, channel_width)

    baseImg = recolor(Escale, COLOR_PALLETE, lvl)
    
    application.canvas.axes = application.canvas.figure.add_subplot()
    application.canvas.axes.imshow(baseImg)
    
    position_x = list(np.arange(0, width + 1, width // 10))
    position_x[-1] = width - 1
    label_x = list(np.arange(int(fMEG - fsMEG//2), int(fMEG + fsMEG//2) + 1, 1))
    label_x = [str(c) for c in label_x]
    
    position_y = list(np.arange(0, height + 1, height // 6))
    position_y[-1] = height - 1
    label_y = list(np.arange(T, -1, -10))
    label_y = [str(c) for c in label_y]
    
    Escale = upscale(E, time_height, channel_width)
    baseImg = recolor(Escale, COLOR_PALLETE, lvl)
    tMid = 0
    for ii in range(T, 360, 1):
        tIn = time.monotonic()
        application.canvas.axes.clear()
        E = np.roll(E, -1, axis = 0)
        E[-1] = x[ii]
        E_new_scale = upscale(np.array([x[ii]]), time_height, channel_width)
        
        addImg = recolor(E_new_scale, COLOR_PALLETE, lvl)       
       
        baseImg = np.roll(baseImg, -1 * time_height, axis = 0)
        baseImg[height - time_height : height] = addImg
        
        application.canvas.axes.imshow(baseImg)

        application.canvas.axes.xaxis.set_major_locator(FixedLocator(position_x))
        application.canvas.axes.xaxis.set_major_formatter(FixedFormatter(label_x))
        application.canvas.axes.xaxis.set_minor_locator(AutoMinorLocator())
        application.canvas.axes.yaxis.set_major_locator(FixedLocator(position_y))
        application.canvas.axes.yaxis.set_major_formatter(FixedFormatter(label_y))
        application.canvas.axes.yaxis.set_minor_locator(AutoMinorLocator())

        application.canvas.draw()
        application.canvas.flush_events()
        tOut = time.monotonic()
        tMid += (tOut-tIn)/(360-T)
    sys.exit(app.exec())