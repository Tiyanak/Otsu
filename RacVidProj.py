import os
from tkFileDialog import askopenfilename
from Tkinter import *
import tkFileDialog
import cv2
import dateutil
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

content = ''
file_path = ''

class Otsu():

    def _init_(self):
        self.rgb = False
        self.regions = 2
        self.img_path = ""
        self.img = None
        self.regR = 2
        self.regG = 2
        self.regB = 2

    def readDataRGB(self, path="", multi=(2, 2, 2)):
        self.rgb = True
        self.regR = multi[0]
        self.regG = multi[1]
        self.regB = multi[2]
        self.img_path = path

    
    def readDataGray(self, path="", multi=2):
        self.rgb = False
        self.regions = multi
        self.img_path = path

    def readImg(self):
        if self.rgb == False:
            self.img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        

    def otsu(self):
        hist = cv2.calcHist([self.img], [0], None, [256], [0,256]) # napravi histogram
        N = np.sum(hist) # Ukupan broj pixela
        Pi = hist.ravel() / N # vjerojatnosti po intenzitetu, Sum(pi) = 1
        Q2 = Pi.cumsum() # kumulativna suma
        bins = np.arange(256)
        vB_max = -np.inf
        thresh = -1
        for i in xrange(1, 256):
            p1, p2 = np.hsplit(Pi, [i]) # vjerojatnosti
            omega1, omega2 = Q2[i], Q2[255]-Q2[i] # kumulativna suma klasa
            b1,b2 = np.hsplit(bins,[i]) # intenziteti klasa
            m1, m2 = np.sum(p1 * b1) / omega1, np.sum(p2 * b2) / omega2 # momenti
            v1,v2 = np.sum(((b1-m1)**2)*p1)/omega1,np.sum(((b2-m2)**2)*p2)/omega2 # varijance
            # trazi maximalan between-class varijancu
            vB = omega1 * omega2 * (m1 - m2) ** 2 
            if vB > vB_max:
                vB_max = vB
                thresh = i

        thresholds = []
        thresholds.append(thresh)
        return (thresholds)

    def buildLookupTable(self, histogram):
        P = np.zeros(shape=(256, 256))
        S = np.zeros(shape=(256, 256))
        H = np.zeros(shape=(256, 256))

        for i in range(1, 256):
            P[i][i] = histogram[i]
            S[i][i] = float(i) * histogram[i]
        
        for i in range(1, 256-1):
            P[1][i + 1] = P[1][i] + histogram[i+1]
            S[1][i + 1] = S[1][i] + float((i+1)) * histogram[i+1]
            
        for i in range(2, 256):
            for j in range(i+1, 256):
                P[i][j] = P[1][j] - P[1][i - 1]
                S[i][j] = S[1][j] - S[1][i - 1]
                
        for i in range(1, 256):
            for j in range(i+1, 256):
                if P[i][j] != 0:
                    H[i][j] = (S[i][j] * S[i][j]) / P[i][j]
                else:
                    H[i][j] = 0.0
                    
        return H

    def multi_otsu(self, img_mat, reg):
        hist = cv2.calcHist([img_mat], [0], None, [256], [0,256]) # napravi histogram
        H = self.buildLookupTable(hist)
        maxSig = -np.inf
        thReg = reg
        thresholds = np.zeros((thReg-1))
        listaMaxova = []
        
        if thReg == 2:
            thresholdsB = [1]
            for i in range(1, 256-thReg):
                Sq = H[1][i] + H[i+1][255]
                listaMaxova.append(Sq)
                if maxSig < Sq and Sq > 0:
                    thresholdsB[0] = i
                    maxSig = Sq
            
            print maxSig
            return thresholdsB

        elif thReg == 3:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    Sq = H[1][i] + H[i + 1][j] + H[j + 1][255]
                    listaMaxova.append(Sq)
                    if maxSig < Sq:
                        thresholds[0] = i
                        thresholds[1] = j
                        maxSig = Sq
            
            print maxSig
            return thresholds
            
        elif thReg == 4:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][255]
                        listaMaxova.append(Sq)
                        if maxSig < Sq:
                            thresholds[0] = i
                            thresholds[1] = j
                            thresholds[2] = k
                            maxSig = Sq
            
            print maxSig
            return thresholds
            
        elif thReg == 5:
            
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][255]
                            if maxSig < Sq:
                                thresholds[0] = i
                                thresholds[1] = j
                                thresholds[2] = k
                                thresholds[3] = l
                                maxSig = Sq
            print maxSig
            return thresholds
            
        elif thReg == 6:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            for m in range(l+1, 256-thReg+4):
                                Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][m] + H[m+1][255]
                                if maxSig < Sq:
                                    thresholds[0] = i
                                    thresholds[1] = j
                                    thresholds[2] = k
                                    thresholds[3] = l
                                    thresholds[4] = m
                                    maxSig = Sq
            print maxSig
            return thresholds
            
        elif thReg == 7:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            for m in range(l+1, 256-thReg+4):
                                for n in range(m+1, 256-thReg+5):
                                    Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][m] + H[m+1][n] + H[n+1][255]
                                    if maxSig < Sq:
                                        thresholds[0] = i
                                        thresholds[1] = j
                                        thresholds[2] = k
                                        thresholds[3] = l
                                        thresholds[4] = m
                                        thresholds[5] = n
                                        maxSig = Sq
            print maxSig    
            return thresholds
            
        elif thReg == 8:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            for m in range(l+1, 256-thReg+4):
                                for n in range(m+1, 256-thReg+5):
                                    for o in range(n+1, 256-thReg+6):
                                        Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][m] + H[m+1][n] + H[n+1][o] + H[o+1][255]
                                        if maxSig < Sq:
                                            thresholds[0] = i
                                            thresholds[1] = j
                                            thresholds[2] = k
                                            thresholds[3] = l
                                            thresholds[4] = m
                                            thresholds[5] = n
                                            thresholds[6] = o
                                            maxSig = Sq
            print maxSig
            return thresholds
        
        elif thReg == 9:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            for m in range(l+1, 256-thReg+4):
                                for n in range(m+1, 256-thReg+5):
                                    for o in range(n+1, 256-thReg+6):
                                        for p in range(o+1, 256-thReg+7):
                                            Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][m] + H[m+1][n] + H[n+1][o] + H[o+1][p] + H[p+1][255]
                                            if maxSig < Sq:
                                                thresholds[0] = i
                                                thresholds[1] = j
                                                thresholds[2] = k
                                                thresholds[3] = l
                                                thresholds[4] = m
                                                thresholds[5] = n
                                                thresholds[6] = o
                                                thresholds[7] = p
                                                maxSig = Sq
            print maxSig
            return thresholds
            
        elif thReg == 10:
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            for m in range(l+1, 256-thReg+4):
                                for n in range(m+1, 256-thReg+5):
                                    for o in range(n+1, 256-thReg+6):
                                        for p in range(o+1, 256-thReg+7):
                                            for q in range(p+1, 256-thReg+8):
                                                Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][m] + H[m+1][n] + H[n+1][o] + H[o+1][p] + H[p+1][q] + H[q+1][255]
                                                if maxSig < Sq:
                                                    thresholds[0] = i
                                                    thresholds[1] = j
                                                    thresholds[2] = k
                                                    thresholds[3] = l
                                                    thresholds[4] = m
                                                    thresholds[5] = n
                                                    thresholds[6] = o
                                                    thresholds[7] = p
                                                    thresholds[8] = q
                                                    maxSig = Sq
            print maxSig
            return thresholds
            
        else:
            thReg = 11
            thresholds = [11]
            for i in range(1, 256-thReg):
                for j in range(i+1, 256-thReg+1):
                    for k in range(j+1, 256-thReg+2):
                        for l in range(k+1, 256-thReg+3):
                            for m in range(l+1, 256-thReg+4):
                                for n in range(m+1, 256-thReg+5):
                                    for o in range(n+1, 256-thReg+6):
                                        for p in range(o+1, 256-thReg+7):
                                            for q in range(p+1, 256-thReg+8):
                                                for r in range(r+1, 256-thReg+9):
                                                    Sq = H[1][i] + H[i + 1][j] + H[j + 1][k] + H[k+1][l] + H[l+1][m] + H[m+1][n] + H[n+1][o] + H[o+1][p] + H[p+1][q] + H[q+1][r] + H[r+1][255]
                                                    if maxSig < Sq:
                                                        thresholds[0] = i
                                                        thresholds[1] = j
                                                        thresholds[2] = k
                                                        thresholds[3] = l
                                                        thresholds[4] = m
                                                        thresholds[5] = n
                                                        thresholds[6] = o
                                                        thresholds[7] = p
                                                        thresholds[8] = q
                                                        thresholds[9] = r
                                                        maxSig = Sq
            print maxSig
            return thresholds

    def pxl_thresh(self, pxl, threshs):
        if len(threshs) > 1:
            for i in range(0, len(threshs)):
                if i == 0:
                    if pxl < threshs[i]:
                        return 0
                    else:
                        continue
                elif i == len(threshs)-1 and threshs[i] <= pxl:
                    return 255
                else:
                    if pxl < threshs[i]:
                        return (threshs[i] + threshs[i-1]) / 2
                    else:
                        continue
        else:
            if pxl < threshs[0]:
                return 0
            else:
                return 255

    def rgb_otsu(self):
        R = self.img[:, :, 0]
        G = self.img[:, :, 1]
        B = self.img[:, :, 2]

        tR = []
        tG = []
        tB = []
   
        tR = self.multi_otsu(R, self.regR)
        tG = self.multi_otsu(G, self.regG)
        tB = self.multi_otsu(B, self.regB)
     
        return (tR, tG, tB)

    def proc_img(self, thresholds):
                
        img_proc = deepcopy(self.img)

        if self.rgb == False:
        
            for i in range(0, len(img_proc)):
                for j in range(0, len(img_proc[0])):
                    img_proc[i, j] = self.pxl_thresh(img_proc[i, j], thresholds)
            return img_proc
        
        else:

            for i in range(0, len(img_proc)):
                for j in range(0, len(img_proc[0])):
                    img_proc[i, j, 0] = self.pxl_thresh(img_proc[i, j, 0], thresholds[0])
                    img_proc[i, j, 1] = self.pxl_thresh(img_proc[i, j, 1], thresholds[1])
                    img_proc[i, j, 2] = self.pxl_thresh(img_proc[i, j, 2], thresholds[2])
                    
            return img_proc

class Application(Frame):
    def __init__(self,master):
        Frame.__init__(self,master)
        self.grid()
        self.create_widgets()
        self.new_dir = "C:/"
        self.file_to_load = []

    def create_widgets(self):
        #select
        Label(self,text="Select properties: RGB/Greyscale",font=(16)).grid(row=1,column=0,sticky=W)
        #checkbutton RGB, Greyscale
        self.RGB=BooleanVar()
        Checkbutton(self,text="RGB",variable=self.RGB).grid(row=2,column=0,sticky=W)
        #number of thresholds
        self.regions = 2
        Label(self,text="Select number of regions: for grayscale").grid(row=2,column=1,sticky=W)
        self.regsGray = Spinbox(self, from_=2, to=10)
        self.regsGray.grid(row=2,column=2,sticky=W)
        Label(self,text="Select number of regions: RED").grid(row=3,column=1,sticky=W)
        self.regsRed = Spinbox(self, from_=2, to=10)
        self.regsRed.grid(row=3,column=2,sticky=W)
        Label(self,text="Select number of regions: GREEN").grid(row=4,column=1,sticky=W)
        self.regsGreen = Spinbox(self, from_=2, to=10)
        self.regsGreen.grid(row=4,column=2,sticky=W)
        Label(self,text="Select number of regions: BLUE").grid(row=5,column=1,sticky=W)
        self.regsBlue = Spinbox(self, from_=2, to=10)
        self.regsBlue.grid(row=5,column=2,sticky=W)
        #image name
        #Label(self,text="Image name (with extension)",font=(16)).grid(row=8,column=0,sticky=W)
        Button(self,text="Choose image (png or jpg)",font=(16),command=self.loadCallBack).grid(row=10,column=0,sticky=W)
        Button(self,text="Choose where to save",font=(16),command=self.saveCallBack).grid(row=10,column=2,sticky=W)
        Button(self,text="Start", font=(16), command=self.run_mul).grid(row=12, column=1, sticky=W)
        Button(self, text="Reset pictures list", font=(16), command=self.reset_files).grid(row=10, column=1, sticky=W)
    
    def loadCallBack(self):
        self.file_to_load.append(tkFileDialog.askopenfilename( filetypes = ( ("JPG files", "*.jpg"), ("PNG files", "*.png"), ("All files", ".*"))))

    def saveCallBack(self):
        self.new_dir = tkFileDialog.askdirectory()

    def reset_files(self):
        self.file_to_load = []

    def run_mul(self):

        for i in range(0, len(self.file_to_load)):
            self.startProgram(self.file_to_load[i])

        print
        print 'DONE'
        print
            

    def startProgram(self, filepath):
        o = Otsu()
        rgb = False
        if self.RGB.get():
            rgb = True
            rgbRegs = (int(self.regsRed.get()), int(self.regsGreen.get()), int(self.regsBlue.get()))
            o.readDataRGB(filepath, rgbRegs)
        else: 
            o.readDataGray(filepath, int(self.regsGray.get()))

        o.readImg()
        img_show = deepcopy(o.img)
        res_img = None

        img_name = filepath.rsplit('/', 1)[-1]
        save_path = self.new_dir + '/' + img_name

        if rgb == True:
            
            cv2.imshow('RGB original', img_show)

            color_threshs = o.rgb_otsu()
            res_img = o.proc_img(color_threshs)

            lista_multi = ', '.join([str(x) for x in color_threshs])
            
            pr = 'OTSU COLOR - thresholds: ' + lista_multi

            cv2.imshow(pr, res_img)
            
            cv2.imwrite(save_path, res_img)

            
        else:
            
            cv2.imshow('GRAYSCALE ORIGINAL',  img_show)
        
            multi_thresh = o.multi_otsu(o.img, int(self.regsGray.get()))
            res_img = o.proc_img(multi_thresh)

            lista_multi = ', '.join([str(x) for x in multi_thresh])
            
            pr = 'OTSU GRAY - THRESHS: ' + lista_multi
           
            cv2.imshow(pr, res_img)

            cv2.imwrite(save_path, res_img)
        
        plt.show()
        
        

root=Tk()
root.title("Otsu Algorithm")
app=Application(root)
root.mainloop()
