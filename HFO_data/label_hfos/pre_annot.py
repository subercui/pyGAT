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
import os

# import sys
# sys.setrecursionlimit(1000000)
# datafile='/home/niking314/Documents/4_data/hfo_classification/rhy/RHY_IID.edf'
# hfofiles=['/home/niking314/Documents/4_data/hfo_classification/rhy/ripple_dets.pik','/home/niking314/Documents/4_data/hfo_classification/rhy/fastripple_dets.pik']

# type='ripple'
# type='fast ripple'

datafile='/home/niking314/Documents/4_data/hfo_classification/liangliwei/liangliwei/SEEG/IID/liangliwei2.edf'
hfofiles=['/home/niking314/Documents/2_Projects/hfo_classification/supervised/subs_results/liangliwei_results/cui_ripple_dets.pik',
          '/home/niking314/Documents/2_Projects/hfo_classification/supervised/subs_results/liangliwei_results/cui_fastripple_dets.pik']

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

hfo_nums=0
for i in hfodets:
    for key,val in i.items():
        hfo_nums+=val.shape[0]

print('hfo nums:',hfo_nums)

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


ripple_annot_results={'hfo_chans':np.array(all_raw_chn_names[0])[np.array(all_hfo_chans[0])],'hfo_times':all_hfo_times[0],'hfo_signals':np.array(all_hfo_signals[0]),'label_dict':['noise','patho','physio'],'sfreq':sfreq}
with open('rippleData.pik','wb') as ripplehandle:
    pik.dump(ripple_annot_results,ripplehandle)



fast_annot_results = {'hfo_chans': np.array(all_raw_chn_names[1])[np.array(all_hfo_chans[1]).astype('int')], 'hfo_times': all_hfo_times[1],
                 'hfo_signals': np.array(all_hfo_signals[1]),
                 'label_dict': ['noise', 'patho', 'physio'],'sfreq':sfreq}
with open('fastRippleData.pik','wb') as fasthandle:
    pik.dump(fast_annot_results,fasthandle)

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



