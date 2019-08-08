import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# from PyQt5.QtWidgets import QApplication,QWidget,QSizePolicy,QGridLayout,QDesktopWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon,QColor
import sys
import pickle as pik
import os
from scipy.signal import butter,filtfilt,get_window
import scipy.signal as sig
from scipy.ndimage import gaussian_filter

def filter_data(data,sfreq,freqband):
    nyq=sfreq/2
    b,a=butter(5,[freqband[0]/nyq,freqband[1]/nyq],btype='bandpass')
    return filtfilt(b,a,data,axis=-1)

class SpectroCanvas(FigureCanvas):
    def __init__(self,parent=None,width=7,height=5,dpi=100):
        # self.figure=Figure(figsize=(width,height),dpi=dpi)
        # self.axes=self.figure.add_subplots
        self.figure,self.axes=plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(width,height),dpi=dpi)
        plt.tight_layout()
        FigureCanvas.__init__(self,self.figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot()

    def plot(self):
        for i in self.axes:
            i.cla()
        self.draw()




class annot_hfo(QWidget):
    def __init__(self):
        super(annot_hfo,self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('annot hfos')
        self.resize(600,1800)
        self.center()
        #main layout
        self.mainLayout=QHBoxLayout()
        self.canvas=SpectroCanvas(self,width=4,height=10)
        self.mainLayout.addWidget(self.canvas)
        #operations layout
        self.workGroupBox=QGroupBox('operations')
        self.mainLayout.addWidget(self.workGroupBox)
        self.workVBoxLayout=QVBoxLayout()
        self.workGroupBox.setLayout(self.workVBoxLayout)
        #files layout
        self.file_box=QGroupBox('file')
        self.workVBoxLayout.addWidget(self.file_box)
        self.file_layout=QGridLayout()
        self.file_box.setLayout(self.file_layout)
        #input data file name
        self.input_file=QLineEdit(self)
        # self.input_file.setText('input file')
        self.input_file.setText('rippleData.pik')
        self.file_layout.addWidget(self.input_file,0,0,1,1)
        #input status
        self.input_status=QLabel(self)
        self.input_status.setText('not loaded')
        self.file_layout.addWidget(self.input_status,0,1,1,1)
        #input data button
        self.input_button=QPushButton('input',self)
        self.file_layout.addWidget(self.input_button,1,0,1,1)
        self.input_button.clicked.connect(self.input_button_func)
        #label file name
        self.output_file=QLineEdit(self)
        # self.output_file.setText('output file')
        self.output_file.setText('test.pik')
        self.file_layout.addWidget(self.output_file,2,0,1,1)
        #label file status
        self.output_status=QLabel(self)
        self.output_status.setText('not loaded')
        self.file_layout.addWidget(self.output_status,2,1,1,1)
        self.hfo_index=0
        #label file new
        self.output_new_button=QPushButton('new',self)
        self.file_layout.addWidget(self.output_new_button,3,0,1,1)
        self.output_new_button.clicked.connect(self.output_new_button_func)
        #label file load
        self.output_load_button=QPushButton('load',self)
        self.file_layout.addWidget(self.output_load_button,3,1,1,1)
        self.output_load_button.clicked.connect(self.output_load_button_func)
        #label file save
        self.output_save_button=QPushButton('save',self)
        self.file_layout.addWidget(self.output_save_button,3,2,1,1)
        self.output_save_button.clicked.connect(self.output_save_button_func)
        #hfo info layout
        self.infoGroupBox=QGroupBox('hfo info')
        self.workVBoxLayout.addWidget(self.infoGroupBox)
        self.infoLayout=QVBoxLayout()
        self.infoGroupBox.setLayout(self.infoLayout)
        #hfo info
        self.hfo_info=QLabel(self)
        self.hfo_info.setText('hfo:(index,channel,class)')
        self.infoLayout.addWidget(self.hfo_info)
        #annot layout
        self.annotGroupBox=QGroupBox('annot buttons')
        self.workVBoxLayout.addWidget(self.annotGroupBox)
        self.annotLayout=QGridLayout()
        self.annotGroupBox.setLayout(self.annotLayout)
        #previous hfo
        self.pre_hfo=QPushButton('prev',self)
        self.annotLayout.addWidget(self.pre_hfo,0,0,1,1)
        self.pre_hfo.clicked.connect(self.pre_hfo_func)
        #next hfo
        self.next_hfo=QPushButton('next',self)
        self.annotLayout.addWidget(self.next_hfo,0,1,1,1)
        self.next_hfo.clicked.connect(self.next_hfo_func)
        #hfo type for freq band disp
        self.switch_ripple_fast=QPushButton('ripple?',self)
        self.annotLayout.addWidget(self.switch_ripple_fast,0,2,1,1)
        self.switch_ripple_fast.clicked.connect(self.switch_ripple_fast_func)
        self.hfoType=True
        self.filter_bank=[80,250]
        self.spec_range=[50,300]
        #patho label
        self.patho_button=QPushButton('patho',self)
        self.annotLayout.addWidget(self.patho_button,1,0,1,1)
        self.patho_button.clicked.connect(self.patho_label_func)
        #physio label
        self.physio_button=QPushButton('physio',self)
        self.annotLayout.addWidget(self.physio_button,1,1,1,1)
        self.physio_button.clicked.connect(self.physio_label_func)
        #noise label
        self.noise_button=QPushButton('noise',self)
        self.annotLayout.addWidget(self.noise_button,1,2,1,1)
        self.noise_button.clicked.connect(self.noise_label_func)
        #skip list
        self.skip_to_list=QListWidget(self)
        # self.skip_to_list.setVerticalScrollBarPolicy()
        self.skip_to_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotLayout.addWidget(self.skip_to_list,2,1,2,1)
        #skip button
        self.skip_to_button=QPushButton('skip',self)
        self.annotLayout.addWidget(self.skip_to_button,4,1,1,1)
        self.skip_to_button.clicked.connect(self.skip_to_func)

        # self.setLayout(self.gridLayout)
        self.setLayout(self.mainLayout)
        self.show()


    def center(self):
        qr=self.frameGeometry()
        cp=QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # def plot(self):
    #     self.canvas.axes.cla()
    #     self.canvas.axes.plot(np.arange(100))
    #     self.canvas.draw()
    #plot hfos
    def draw_hfo(self):
        #self.hfo_index
        self.current_hfo_signal=self.hfoData['hfo_signals'][self.hfo_index]
        half_width=int(len(self.current_hfo_signal)/2)
        self.canvas.axes[0].cla()
        self.canvas.axes[0].plot(np.arange(len(self.current_hfo_signal))/ self.hfoData['sfreq'], self.current_hfo_signal)
        self.canvas.axes[0].set_title('raw signal')
        self.canvas.axes[0].set_yticks([])
        self.canvas.axes[1].cla()
        self.canvas.axes[1].plot(np.arange(len(self.current_hfo_signal))/ self.hfoData['sfreq'],
                     filter_data(self.current_hfo_signal,self.hfoData['sfreq'], [self.filter_bank[0], self.filter_bank[1]]))
        self.canvas.axes[1].set_title('%d-%d HZ' % (self.filter_bank[0], self.filter_bank[1]))
        self.canvas.axes[1].set_yticks([])
        self.canvas.axes[2].cla()
        f, t, hfo_spec = sig.spectrogram(self.current_hfo_signal, fs=self.hfoData['sfreq'],window=get_window('hann',int(0.7*half_width)),nperseg=int(0.7 * half_width),
                                         noverlap=int(0.95 * 0.7 * half_width), nfft=2000, mode='magnitude')
        hfo_spec = gaussian_filter(hfo_spec, sigma=2)
        spec_x, spec_y = np.meshgrid(t, f[self.spec_range[0]:self.spec_range[1]])
        self.canvas.axes[2].pcolor(spec_x, spec_y, hfo_spec[self.spec_range[0]:self.spec_range[1]], cmap='jet')
        self.canvas.axes[2].set_title('spectrogram')
        self.canvas.axes[2].set_xlabel('time [s]')
        self.canvas.draw()

        self.status_text='{} band ({}, {}, {})'.format('ripple' if self.hfoType else 'fast ripple',self.hfo_index,self.hfoData['hfo_chans'][self.hfo_index],self.hfoData['label_dict'][int(self.hfoLabels[self.hfo_index])])
        self.hfo_info.setText(self.status_text)

        self.skip_to_list.clear()
        self.skip_to_list.addItems(list(np.arange(len(self.hfoLabels)).astype('str')))
        for i,j in enumerate(self.hfoLabels):
            if j==1:
                self.skip_to_list.item(i).setBackground(QColor('red'))
            elif j==2:
                self.skip_to_list.item(i).setBackground(QColor('blue'))
            elif j==0:
                self.skip_to_list.item(i).setBackground(QColor('white'))



    #input and ouput funcs
    def input_button_func(self):
        inputFileName=self.input_file.text()
        if os.path.exists(inputFileName):
            with open(inputFileName,'rb') as inputFileHandle:
                self.hfoData=pik.load(inputFileHandle)
            self.input_status.setText('loaded')
        else:
            self.input_file.setText('not right file name')

    def output_new_button_func(self):
        self.hfoLabels=np.zeros(self.hfoData['hfo_signals'].shape[0])
        self.output_status.setText('newed')
        self.hfo_index=0
        self.draw_hfo()

    def output_load_button_func(self):
        labelFileName=self.output_file.text()
        if os.path.exists(labelFileName):
            with open(labelFileName,'rb') as labelFileHandle:
                self.hfoLabels=pik.load(labelFileHandle)
            self.output_status.setText('loaded')
            self.hfo_index=0
            self.draw_hfo()
        else:
            self.output_file.setText('not right file name')

    def output_save_button_func(self):
        labelFileName=self.output_file.text()
        with open(labelFileName,'wb') as labelFileHandle:
            pik.dump(self.hfoLabels,labelFileHandle)
        self.output_status.setText('saved')

    #label edit funcs
    def pre_hfo_func(self):
        if self.hfo_index>=1:
            self.hfo_index-=1

        self.draw_hfo()

    def next_hfo_func(self):
        if self.hfo_index<len(self.hfoLabels)-1:
            self.hfo_index+=1
        self.draw_hfo()

    def switch_ripple_fast_func(self):
        self.hfoType=not self.hfoType
        if self.hfoType:
            self.filter_bank=[80,250]
            self.spec_range=[50,300]
        else:
            self.filter_bank=[250,500]
            self.spec_range=[220,500]
        self.draw_hfo()

    def patho_label_func(self):
        self.hfoLabels[self.hfo_index]=1
        self.skip_to_list.item(self.hfo_index).setBackground(QColor('red'))
        self.status_text = '{} band ({}, {}, {})'.format('ripple' if self.hfoType else 'fast ripple', self.hfo_index,
                                                         self.hfoData['hfo_chans'][self.hfo_index],
                                                         self.hfoData['label_dict'][
                                                             int(self.hfoLabels[self.hfo_index])])
        self.hfo_info.setText(self.status_text)

    def physio_label_func(self):
        self.hfoLabels[self.hfo_index]=2
        self.skip_to_list.item(self.hfo_index).setBackground(QColor('blue'))
        self.status_text = '{} band ({}, {}, {})'.format('ripple' if self.hfoType else 'fast ripple', self.hfo_index,
                                                         self.hfoData['hfo_chans'][self.hfo_index],
                                                         self.hfoData['label_dict'][
                                                             int(self.hfoLabels[self.hfo_index])])
        self.hfo_info.setText(self.status_text)

    def noise_label_func(self):
        self.hfoLabels[self.hfo_index]=0
        self.skip_to_list.item(self.hfo_index).setBackground(QColor('white'))
        self.status_text = '{} band ({}, {}, {})'.format('ripple' if self.hfoType else 'fast ripple', self.hfo_index,
                                                         self.hfoData['hfo_chans'][self.hfo_index],
                                                         self.hfoData['label_dict'][
                                                             int(self.hfoLabels[self.hfo_index])])
        self.hfo_info.setText(self.status_text)

    #skip to
    def skip_to_func(self):
        if len(self.skip_to_list.selectedItems())>0:
            self.selected_hfo=self.skip_to_list.selectedItems()[0]
        self.hfo_index=int(self.selected_hfo.text())
        self.draw_hfo()


if __name__=='__main__':
    app=QApplication(sys.argv)
    annoter=annot_hfo()
    sys.exit(app.exec_())
