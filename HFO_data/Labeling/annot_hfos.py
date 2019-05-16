#for ripple and fastripple
import mne
from mne.io.constants import FIFF
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import pickle as pik
import re
from scipy.ndimage import gaussian_filter



#datafile='/home/niking314/Documents/4_data/hfo_classification/rhy/RHY_IID.edf'
#hfofiles=['/home/niking314/Documents/4_data/hfo_classification/rhy/ripple_dets.pik','/home/niking314/Documents/4_data/hfo_classification/rhy/fastripple_dets.pik']
datafile='D:\HFO_dev\IID_Data\RHY_IID.edf'
hfofiles=['D:\HFO_dev\IID_Data\RHYFastRipple.pik','D:\HFO_dev\IID_Data\RHYripple.pik']
#type='ripple'
type='fast ripple'

plotgap=3e-3

def filter_data(data,low,high,fs):
    nyq=fs/2
    b,a=scipy.signal.butter(5,np.array([low/nyq,high/nyq]),btype='bandpass')
    return scipy.signal.filtfilt(b,a,data)

def pad_zero(data,length):
    data_len=len(data)
    if data_len<length:
        tmp_data=np.zeros(length)
        tmp_data[:data_len]=data
        return tmp_data
    return data

def correct_kind(raw):
    chs = raw.info['chs']
    for i in range(len(chs)):
        kind = chs[i]['kind']
        name = chs[i]['ch_name']
        if kind is FIFF.FIFFV_EEG_CH:
            if (u'STIM' in name) or (u'stim' in name):
                chs[i]['kind'] = FIFF.FIFFV_STIM_CH
                continue
            elif (u'EOG' in name) or (u'eog' in name):
                chs[i]['kind'] = FIFF.FIFFV_EOG_CH
                continue
            elif (u'EMG' in name) or (u'emg' in name):
                chs[i]['kind'] = FIFF.FIFFV_EMG_CH
                continue
            elif (u'ECG' in name) or (u'ecg' in name):
                chs[i]['kind'] = FIFF.FIFFV_ECG_CH
                continue
            elif (u'MISC' in name) or (u'misc' in name):
                chs[i]['kind'] = FIFF.FIFFV_MISC_CH
                continue
            elif (u'DC' in name) or (u'dc' in name):
                chs[i]['kind'] = FIFF.FIFFV_SYST_CH
                continue
    raw.info['chs'] = chs
    return

raw_data=mne.io.read_raw_edf(datafile,preload=False,stim_channel=None)
correct_kind(raw_data)
sfreq=raw_data.info['sfreq']

hfodets=[]
for i in hfofiles:
    with open(i,'rb') as hfohandle:
        hfodets.append(pik.load(hfohandle))

for i in hfodets:
    not_real_chans=[]
    for key,val in i.items():
        if not re.search(r"(\w+) (\w)('?)(\d+)",key):
            not_real_chans.append(key)
    for j in not_real_chans:
        del i[j]


half_width=int(128e-3*sfreq)

all_raw_signal_times=[]
all_raw_signals=[]
all_raw_chn_names=[]
all_hfo_times=[]
all_hfo_signals=[]
all_hfo_chans=[]
all_hfo_xy=[]
all_chn_nums=[]

for dets in hfodets:

    raw_signals=[]
    raw_chn_names=[]
    hfo_times=[]
    hfo_signals=[]
    hfo_chans=[]
    hfo_xy=[]
    #plot raw signal
    n_chans=len(dets)
    all_chn_nums.append(n_chans)
    for i,(ch,val) in enumerate(dets.items()):
        raw_chn_names.append(ch)
        ch_idx=raw_data.info['ch_names'].index(ch)
        raw_signal,tmp_time=raw_data[ch_idx,:]
        if i==0:
            raw_signal_time=tmp_time
            all_raw_signal_times.append(raw_signal_time)
        raw_signals.append(raw_signal.ravel())
        for hfo in val:
            hfo_start, hfo_end = raw_data.time_as_index([hfo[0], hfo[1]])
            hfo_middle = int((hfo_start + hfo_end) / 2)
            if hfo_middle-half_width<0:continue
            hfo_signal,hfo_time = raw_data[ch_idx, hfo_middle - half_width:hfo_middle + half_width]
            hfo_signal = pad_zero(hfo_signal.ravel(), 2 * half_width)
            hfo_times.append(hfo_time)
            hfo_signals.append(hfo_signal)
            hfo_chans.append(i)
            hfo_xy.append([(hfo[0]+hfo[1])/2,plotgap*(n_chans-1-i)+hfo_signal.mean()])
    all_raw_signals.append(raw_signals)
    all_raw_chn_names.append(raw_chn_names)
    all_hfo_times.append(hfo_times)
    all_hfo_signals.append(hfo_signals)
    all_hfo_chans.append(hfo_chans)
    all_hfo_xy.append(hfo_xy)


class plot_hfo_spec(object):
    def __init__(self,fig_canvas,ax,hfo_times,hfo_signals,hfo_chans,raw_chn_names,raw_chn_nums,raw_chn_signal_times,raw_chn_signals,hfo_xy,filter_bank='ripple',annot_labels=None,annot_stage='pathological'):
        self.figcanvas=fig_canvas
        self.ax=ax
        self.hfo_times=hfo_times
        self.hfo_signals=hfo_signals
        self.hfo_chans=hfo_chans
        self.raw_chn_names=raw_chn_names
        self.raw_chn_nums=raw_chn_nums
        self.raw_chn_signal_times=raw_chn_signal_times
        self.raw_chn_signals=raw_chn_signals
        self.hfo_xy=hfo_xy
        self.annot_labels=annot_labels
        self.annot_stage=annot_stage
        self.color_sch=['k','r','b']
        assert len(self.hfo_times)==self.annot_labels.shape[0],'wrong annot length'
        if filter_bank=='ripple':
            self.filter_bank=[80.,250.]
            self.spec_range=[50,300]
            self.ripple_type='ripple'
        elif filter_bank=='fast ripple':
            self.filter_bank=[250.,500.]
            self.spec_range=[220,500]
            self.ripple_type='fast ripple'
        self.figcanvas.mpl_connect('button_press_event',self.button_press_func)

    def button_press_func(self,e):
        # print('pressed')
        if e.button==3:
            self.press_x=e.xdata
            self.press_y=e.ydata
            print(self.press_x,self.press_y)
            self.distance=(np.array(self.hfo_xy)-np.array((self.press_x,self.press_y)))**2
            self.distance=self.distance/np.std(self.distance,axis=0)
            self.distance=np.sum(self.distance,axis=1)
            self.press_idx=np.argmin(self.distance)
            # print(np.array(self.hfo_xy)[self.press_idx])
            # print(self.press_idx)
            self.press_hfo_time=self.hfo_times[self.press_idx]
            self.press_hfo_signal=self.hfo_signals[self.press_idx]
            self.press_hfo_chn_name=self.raw_chn_names[self.hfo_chans[self.press_idx]]

            # plt.figure(self.press_hfo_chn_name+' '+self.ripple_type,figsize=(1.5,6))
            plt.figure(self.ripple_type,figsize=(1.5,6))
            ax_orig=plt.subplot(3,1,1)
            ax_orig.cla()
            ax_orig.plot(self.press_hfo_time[0]+np.arange(self.press_hfo_time.shape[0])/sfreq,self.press_hfo_signal)
            ax_orig.set_title('raw signal')
            ax_orig.set_yticks([])
            ax_filt=plt.subplot(3,1,2,sharex=ax_orig)
            ax_filt.cla()
            ax_filt.plot(self.press_hfo_time[0]+np.arange(self.press_hfo_time.shape[0])/sfreq,filter_data(self.press_hfo_signal,self.filter_bank[0],self.filter_bank[1],sfreq))
            ax_filt.set_title('%d-%d HZ'%(self.filter_bank[0],self.filter_bank[1]))
            ax_filt.set_yticks([])
            ax_spec=plt.subplot(3,1,3,sharex=ax_orig)
            ax_spec.cla()
            f,t,hfo_spec=sig.spectrogram(self.press_hfo_signal,fs=sfreq,nperseg=int(0.7*half_width),noverlap=int(0.95*0.7*half_width),nfft=2000,mode='magnitude')
            hfo_spec=gaussian_filter(hfo_spec,sigma=2)
            spec_x,spec_y=np.meshgrid(t,f[self.spec_range[0]:self.spec_range[1]])
            spec_x=spec_x+self.press_hfo_time[0]
            ax_spec.pcolor(spec_x,spec_y,hfo_spec[self.spec_range[0]:self.spec_range[1]],cmap='jet')
            ax_spec.set_title('spectrogram')
            ax_spec.set_xlabel('time [s]')

            plt.tight_layout()
            plt.show()
        elif e.button==1:
            self.press_x=e.xdata
            self.press_y=e.ydata
            print(self.press_x,self.press_y)
            self.distance=(np.array(self.hfo_xy)-np.array((self.press_x,self.press_y)))**2
            self.distance=self.distance/np.std(self.distance,axis=0)
            self.distance=np.sum(self.distance,axis=1)
            self.press_idx=np.argmin(self.distance)
            if self.annot_stage=='pathological':
                if self.annot_labels[self.press_idx]==0:
                    self.annot_labels[self.press_idx]=1
                elif self.annot_labels[self.press_idx]==1:
                    self.annot_labels[self.press_idx]=0
            elif self.annot_stage=='physiological':
                if self.annot_labels[self.press_idx]==0:
                    self.annot_labels[self.press_idx]=2
                elif self.annot_labels[self.press_idx]==2:
                    self.annot_labels[self.press_idx]=0
            self.refresh_plot()

    def refresh_plot(self):
        self.ax.cla()
        for i in range(self.raw_chn_nums):
            self.ax.plot(self.raw_chn_signal_times,self.raw_chn_signals[i]+plotgap*(self.raw_chn_nums-1-i),color='#808080',alpha=0.5)
            self.ax.text(0,plotgap*(self.raw_chn_nums-1-i),self.raw_chn_names[i],horizontalalignment='right',verticalalignment='center',color='k')#'r' if i%2==0 else 'b')
        for i in range(len(self.hfo_chans)):
            self.ax.plot(self.hfo_times[i],self.hfo_signals[i]+plotgap*(self.raw_chn_nums-1-self.hfo_chans[i]),color=self.color_sch[int(self.annot_labels[i])])
        self.ax.axis(xmin=0,ymin=-plotgap/2,ymax=plotgap*(self.raw_chn_nums-0.5))
        self.ax.set_title(self.ripple_type+' detection')
        self.ax.set_xlabel('time [s]')
        self.ax.set_yticks([])
        self.figcanvas.draw()



if type=='ripple':
    color_sch=['k','r','b']
    ripple_fig=plt.figure('ripple stage 1')
    ripple_ax=ripple_fig.add_subplot(111)
    ripple_ax.set_yticks([])
    for i in range(len(all_raw_chn_names[0])):
        ripple_ax.plot(all_raw_signal_times[0],all_raw_signals[0][i]+plotgap*(all_chn_nums[0]-1-i),color='#808080',alpha=0.5)#'='r' if i%2==0 else 'b')
        ripple_ax.text(0,plotgap*(all_chn_nums[0]-1-i),all_raw_chn_names[0][i],horizontalalignment='right',verticalalignment='center',color='k')#'r' if i%2==0 else 'b')
    for i in range(len(all_hfo_chans[0])):
        ripple_ax.plot(all_hfo_times[0][i],all_hfo_signals[0][i]+plotgap*(all_chn_nums[0]-1-all_hfo_chans[0][i]),color='k') #if all_hfo_chans[0][i]%2==0 else 'b')
    ripple_ax.axis(xmin=0,ymin=-plotgap/2,ymax=plotgap*(all_chn_nums[0]-0.5))
    ripple_ax.set_title('ripple detection')
    ripple_ax.set_xlabel('time [s]')
    annot_labels=np.zeros(shape=len(all_hfo_chans[0]))
    ripple_show_hfo=plot_hfo_spec(ripple_fig.canvas,ripple_ax,all_hfo_times[0],all_hfo_signals[0],all_hfo_chans[0],all_raw_chn_names[0],all_chn_nums[0],all_raw_signal_times[0],all_raw_signals[0],all_hfo_xy[0],'ripple',annot_labels=annot_labels,annot_stage='pathological')
    plt.show()
    #replot for stage 2
    ripple_fig = plt.figure('ripple stage 2')
    ripple_ax = ripple_fig.add_subplot(111)
    ripple_ax.set_yticks([])
    for i in range(len(all_raw_chn_names[0])):
        ripple_ax.plot(all_raw_signal_times[0], all_raw_signals[0][i] + plotgap * (all_chn_nums[0] - 1 - i),
                       color='#808080', alpha=0.5)  # '='r' if i%2==0 else 'b')
        ripple_ax.text(0, plotgap * (all_chn_nums[0] - 1 - i), all_raw_chn_names[0][i], horizontalalignment='right',
                       verticalalignment='center', color='k')#'r' if i % 2 == 0 else 'b')
    for i in range(len(all_hfo_chans[0])):
        ripple_ax.plot(all_hfo_times[0][i],
                       all_hfo_signals[0][i] + plotgap * (all_chn_nums[0] - 1 - all_hfo_chans[0][i]),
                       color=color_sch[int(annot_labels[i])])  # if all_hfo_chans[0][i]%2==0 else 'b')
    ripple_ax.axis(xmin=0, ymin=-plotgap / 2, ymax=plotgap * (all_chn_nums[0] - 0.5))
    ripple_ax.set_title('ripple detection')
    ripple_ax.set_xlabel('time [s]')
    ripple_show_hfo=plot_hfo_spec(ripple_fig.canvas,ripple_ax,all_hfo_times[0],all_hfo_signals[0],all_hfo_chans[0],all_raw_chn_names[0],all_chn_nums[0],all_raw_signal_times[0],all_raw_signals[0],all_hfo_xy[0],'ripple',annot_labels=annot_labels,annot_stage='physiological')
    plt.show()
    annot_results={'hfo_chans':np.array(all_raw_chn_names[0])[np.array(all_hfo_chans[0])],'hfo_times':all_hfo_times[0],'hfo_signals':all_hfo_signals[0],'hfo_labels':annot_labels,'label_dict':['noise','patho','physio']}
    with open('rippleLabels.pik','wb') as ripplehandle:
        pik.dump(annot_results,ripplehandle)


else:
    color_sch=['k','r','g']
    fastripple_fig=plt.figure('fast ripple stage 1')
    fastripple_ax=fastripple_fig.add_subplot(111)
    fastripple_ax.set_yticks([])
    for i in range(len(all_raw_chn_names[1])):
        fastripple_ax.plot(all_raw_signal_times[1],all_raw_signals[1][i]+plotgap*(all_chn_nums[1]-1-i),color='#808080',alpha=0.5)#'='r' if i%2==0 else 'b')
        fastripple_ax.text(0,plotgap*(all_chn_nums[1]-1-i),all_raw_chn_names[1][i],horizontalalignment='right',verticalalignment='center',color='k')#'r' if i%2==0 else 'b')
    for i in range(len(all_hfo_chans[1])):
        fastripple_ax.plot(all_hfo_times[1][i],all_hfo_signals[1][i]+plotgap*(all_chn_nums[1]-1-all_hfo_chans[1][i]),color='k') #if all_hfo_chans[1][i]%2==0 else 'b')
    fastripple_ax.axis(xmin=0,ymin=-plotgap/2,ymax=plotgap*(all_chn_nums[1]-0.5))
    fastripple_ax.set_title('fast ripple detection')
    fastripple_ax.set_xlabel('time [s]')
    annot_labels=np.zeros(shape=len(all_hfo_chans[1]))
    fastripple_show_hfo=plot_hfo_spec(fastripple_fig.canvas,fastripple_ax,all_hfo_times[1],all_hfo_signals[1],all_hfo_chans[1],all_raw_chn_names[1],all_chn_nums[1],all_raw_signal_times[1],all_raw_signals[1],all_hfo_xy[1],'fast ripple',annot_labels=annot_labels,annot_stage='pathological')
    plt.show()
    #replot for stage2
    fastripple_fig = plt.figure('fast ripple stage 2')
    fastripple_ax = fastripple_fig.add_subplot(111)
    fastripple_ax.set_yticks([])
    for i in range(len(all_raw_chn_names[1])):
        fastripple_ax.plot(all_raw_signal_times[1], all_raw_signals[1][i] + plotgap * (all_chn_nums[1] - 1 - i),
                           color='#808080', alpha=0.5)  # '='r' if i%2==0 else 'b')
        fastripple_ax.text(0, plotgap * (all_chn_nums[1] - 1 - i), all_raw_chn_names[1][i], horizontalalignment='right',
                           verticalalignment='center', color='k')#'r' if i % 2 == 0 else 'b')
    for i in range(len(all_hfo_chans[1])):
        fastripple_ax.plot(all_hfo_times[1][i],
                           all_hfo_signals[1][i] + plotgap * (all_chn_nums[1] - 1 - all_hfo_chans[1][i]),
                           color=color_sch[int(annot_labels[i])])  # if all_hfo_chans[1][i]%2==0 else 'b')
    fastripple_ax.axis(xmin=0, ymin=-plotgap / 2, ymax=plotgap * (all_chn_nums[1] - 0.5))
    fastripple_ax.set_title('fast ripple detection')
    fastripple_ax.set_xlabel('time [s]')
    fastripple_show_hfo = plot_hfo_spec(fastripple_fig.canvas, fastripple_ax, all_hfo_times[1], all_hfo_signals[1],
                                        all_hfo_chans[1], all_raw_chn_names[1], all_chn_nums[1], all_raw_signal_times[1],
                                        all_raw_signals[1], all_hfo_xy[1], 'fast ripple', annot_labels=annot_labels,
                                        annot_stage='physiological')
    plt.show()
    annot_results = {'hfo_chans': np.array(all_raw_chn_names[1])[np.array(all_hfo_chans[1]).astype('int')], 'hfo_times': all_hfo_times[1],
                     'hfo_signals': all_hfo_signals[1], 'hfo_labels': annot_labels,
                     'label_dict': ['noise', 'patho', 'physio']}
    with open('fastRippleLabels.pik','wb') as fasthandle:
        pik.dump(annot_results,fasthandle)

# plt.show()




#
#
# def plot_hfo(hfodets,raw_chn_names,raw_signal_time,raw_signals,hfo_times,hfo_signals,hfo_chans,hfo_xy,filter_bank=[80.,250.],plot_name='hfo_dets'):
#     fig=plt.figure(plot_name)
#     ax=fig.add_subplot(111)
#     ax.set_yticks([])
#     for i in range(len(raw_chn_names)):
#         ax.plot(raw_signal_time,raw_signals[i]+plotgap*(n_chans-1-i),color='k')#'='r' if i%2==0 else 'b')
#         ax.text(0,plotgap*(n_chans-1-i),raw_chn_names[i],horizontalalignment='right',verticalalignment='center',color='r' if i%2==0 else 'b')
#     for i in range(len(hfo_chans)):
#         ax.plot(hfo_times[i],hfo_signals[i]+plotgap*(n_chans-1-hfo_chans[i]),color='r' if hfo_chans[i]%2==0 else 'b')
#     show_hfo=plot_hfo_spec(fig.canvas,ax,hfo_times,hfo_signals,hfo_chans,raw_chn_names,hfo_xy,filter_bank)
#
#
# plot_hfo(hfodets,raw_chn_names,raw_signal_time,raw_signals,hfo_times,hfo_signals,hfo_chans,hfo_xy,filter_bank=[80.,250.],plot_name='ripple')
#
# plt.show()






#data to plot hfo
#hfo time
#hfo signal
#hfo channel index
#hfo time and amp middle point for show spectrogram



