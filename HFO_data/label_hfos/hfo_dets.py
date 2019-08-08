# -*- coding: utf-8 -*-
#save hfo detections
import numpy as np
from scipy import signal 
import mne
from mne.io.constants import FIFF
from matplotlib import pyplot as plt
from matplotlib import patches
import time
import csv
import pickle as pik

#/home/niking314/Documents/4_data/yuquan/RHY_IID.edf

def find_HFO(raw, bands=[(80.,250.),(250.,500.)], abs_thresholds=[10.,5.], rel_thresholds=[2.,2.], picks='EEG', plot_names=None, start_time = 0):
    time_in = time.time()
    if picks is 'EEG':
        correct_kind(raw)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    pick_names = np.array(raw.info['ch_names'])[picks]
    raws = []
    peak_inds = []
    gaps = []
    thresholds = []
    large_peak_inds = []
    HFO_peak_inds = []
    HFO_times = []
    HFO_amps = []
    for i in range(len(bands)):
        temp_raw = raw.copy()
        temp_raw.filter(bands[i][0], bands[i][1], picks=picks, fir_design='firwin')
        data, times = temp_raw[picks]
        peak_ind = data_maxima_ind(data)
        stds = 2*np.std(data, axis=1)
        threshold = np.max(np.vstack((rel_thresholds[i]*stds, 0*stds+abs_thresholds[i]*1e-6)), axis=0)
        large_peak_ind = []
        large_peak_P_ind = []
        large_peak_N_ind = []
        for j in range(len(picks)):
#            large_peak_ind.append(np.where(abs(data[j,peak_ind[j]])>=threshold[j])[0])
            large_peak_P_ind.append(np.where(data[j, peak_ind[j]] >= threshold[j])[0])
            large_peak_N_ind.append(np.where(data[j, peak_ind[j]] <= -threshold[j])[0])
            large_peak_ind.append(np.sort(np.hstack((large_peak_P_ind[-1], large_peak_N_ind[-1]))))
        HFO_peak_inds_this_band, HFO_amps_this_band = locate_HFO(data, peak_ind, large_peak_P_ind, large_peak_N_ind)
        HFO_times_this_band = []
        for j in range(len(picks)):
            HFO_time_this_band = []
            for k in range(len(HFO_peak_inds_this_band[j])):
                HFO_time_interval = np.round(times[peak_ind[j][HFO_peak_inds_this_band[j][k]]], 2)
                HFO_time_interval = [x + start_time for x in HFO_time_interval]
                HFO_time_this_band.append(HFO_time_interval)
            HFO_times_this_band.append(HFO_time_this_band)
        gap = 10 * int(np.max(2. * threshold * 1e6) / 10.) + 10
        raws.append(temp_raw)
        peak_inds.append(peak_ind)
        gaps.append(gap)
        thresholds.append(threshold)
        large_peak_inds.append(large_peak_ind)
        HFO_peak_inds.append(HFO_peak_inds_this_band)
        HFO_times.append(HFO_times_this_band)
        HFO_amps.append(HFO_amps_this_band)
    time_out = time.time()
    print ('Done in %.3f secounds.'%(time_out-time_in))
    
    if plot_names is not None:
        for i in range(len(bands)):
            plot_data(raws[i], picks, plot_names, gaps[i], start=29.6, stop=29.8, peak_ind=peak_inds[i], threshold=thresholds[i], large_peak_ind=large_peak_inds[i], HFO_peak_ind=HFO_peak_inds[i], show_detail=True)

    return HFO_times, HFO_amps, picks, pick_names, raws
    
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

def further_pick_by_name(raw, picks, channel_names):
    further_picks = []
    all_names = np.array(raw.info['ch_names'])[picks]
    for i in range(len(channel_names)):
        this_name = channel_names[i]
        for j in range(len(all_names)):
            if this_name in all_names[j]:
                further_picks.append(j)
    return np.array(further_picks)

def data_maxima_ind(data):
    n_picks = data.shape[0]
    maxima_inds = []
    for i in range(n_picks):
        maxima_P_ind = np.array(signal.argrelmax(data[i,:]))[0]
        maxima_N_ind = np.array(signal.argrelmin(data[i,:]))[0]
        maxima_ind = sorted(maxima_P_ind.tolist() + maxima_N_ind.tolist())
        if len(maxima_ind) == 0:
            maxima_ind.append(0)
        maxima_inds.append(np.array(maxima_ind))
    return maxima_inds

def locate_HFO(data, peak_ind, large_peak_P_ind, large_peak_N_ind, min_osc=4):
    HFO_peak_inds = []
    HFO_amps = []
    for i_picks in range(len(large_peak_N_ind)):
        HFO_peak_ind = []
        HFO_amp = []
        done_N = np.zeros(len(large_peak_N_ind[i_picks]))
        for i_N in range(len(large_peak_N_ind[i_picks])):
            if not done_N[i_N]:
                count_osc_N = 0
                while i_N+count_osc_N+1 < len(large_peak_N_ind[i_picks]):
                    if large_peak_N_ind[i_picks][i_N+count_osc_N+1] == large_peak_N_ind[i_picks][i_N+count_osc_N]+2:
                        count_osc_N += 1
                    else:
                        break
                if count_osc_N < min_osc-1:
                    done_N[i_N:i_N+count_osc_N+1] = np.ones(count_osc_N+1)
                    continue
                else:
                    i_N_start = large_peak_N_ind[i_picks][i_N]
                    i_N_end = large_peak_N_ind[i_picks][i_N+count_osc_N]
                    left_P_ind = np.where(large_peak_P_ind[i_picks]==i_N_start-1)[0]
                    if len(left_P_ind) > 0:
                        i_P = left_P_ind[0]
                    else:
                        right_P_ind = np.where(large_peak_P_ind[i_picks]==i_N_start+1)[0]
                        if len(right_P_ind) > 0:
                            i_P = right_P_ind[0]
                        else:
                            continue
                    count_osc_P = 0
                    while i_P+count_osc_P+1 < len(large_peak_P_ind[i_picks]):
                        if large_peak_P_ind[i_picks][i_P+count_osc_P+1] == large_peak_P_ind[i_picks][i_P+count_osc_P]+2:
                            count_osc_P += 1
                        else:
                            break
                    if count_osc_P < min_osc-1:
                        continue
                    else:
                        i_P_start = large_peak_P_ind[i_picks][i_P]
                        i_P_end = min([large_peak_P_ind[i_picks][i_P+count_osc_P],i_N_end+1])
                        i_HFO_start = min([i_N_start, i_P_start])
                        if abs(i_P_end-i_N_end) <= 1:
                            i_HFO_end = max([i_N_end, i_P_end])
                        else:
                            i_HFO_end = i_P_end + 1
                        count_done = 0
                        while i_N+count_done<len(large_peak_N_ind[i_picks]) and large_peak_N_ind[i_picks][i_N+count_done]<=i_HFO_end:
                            done_N[i_N+count_done] = True        
                            count_done += 1
                        HFO_peak_ind.append(np.array([i_HFO_start, i_HFO_end]))
                        this_amp = abs(data[i_picks][peak_ind[i_picks][i_HFO_start:i_HFO_end+1]]).mean()
                        HFO_amp.append(1e6*this_amp)
        HFO_peak_inds.append(HFO_peak_ind)
        HFO_amps.append(HFO_amp)
    return HFO_peak_inds, HFO_amps

def plot_data(raw, picks, plot_names, gap=1000, start=0., stop=None, peak_ind=None, threshold=None, large_peak_ind=None, HFO_peak_ind=None, show_detail=False, HFO_time=None, line_colors=['b']):
    further_picks = further_pick_by_name(raw, picks, plot_names)    
    temp_raw = raw.copy()
    temp_raw.crop(start, stop)
    data, times = temp_raw[picks[further_picks]]
    sfreq = raw.info['sfreq']
    names = np.array(raw.info['ch_names'])[picks[further_picks]]
    data = data*1e6
    n_plots = len(further_picks)
    if show_detail and (peak_ind is not None):
        mark_inds = []
        for i in range(n_plots):
            mark_ind = peak_ind[further_picks[i]]-(temp_raw.first_samp-raw.first_samp)
            mark_ind = mark_ind[np.where(mark_ind>=0)]
            mark_ind = mark_ind[np.where(mark_ind<data.shape[1])]
            mark_inds.append(mark_ind)
        if large_peak_ind is not None:
            large_mark_inds = []
            for i in range(n_plots):
                large_mark_ind = peak_ind[further_picks[i]][large_peak_ind[further_picks[i]]]-(temp_raw.first_samp-raw.first_samp)
                large_mark_ind = large_mark_ind[np.where(large_mark_ind>=0)]
                large_mark_ind = large_mark_ind[np.where(large_mark_ind<data.shape[1])]
                large_mark_inds.append(large_mark_ind)
        if HFO_peak_ind is not None:
            HFO_mark_inds = []
            for i in range(n_plots):
                temp_HFO_ind = np.hstack(HFO_peak_ind[further_picks[i]]) if len(HFO_peak_ind[further_picks[i]])>0 else []
                HFO_mark_ind = peak_ind[further_picks[i]][temp_HFO_ind]-(temp_raw.first_samp-raw.first_samp)
                HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind>=0)]
                HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind<data.shape[1])]
                HFO_mark_inds.append(HFO_mark_ind)
    if (not show_detail) and (HFO_time is not None):
        HFO_mark_inds = []
        for i in range(n_plots):
            temp_HFO_ind = (sfreq*np.hstack(HFO_time[further_picks[i]])).astype(int) if len(HFO_time[further_picks[i]])>0 else []
            HFO_mark_ind = temp_HFO_ind - (temp_raw.first_samp-raw.first_samp)
            HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind>=0)]
            HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind<data.shape[1])]
            HFO_mark_inds.append(HFO_mark_ind)

    plt.figure(figsize=[14, 1.*n_plots])
    data_plot = plt.gca()
    if stop is None:
        stop = start+times[-1]
    data_plot.axis(xmin=start, xmax=stop, ymin=-0.5*gap, ymax=(n_plots-0.5)*gap)
    data_plot.set_yticks([]) 
    for i in range(n_plots):
        line_color = line_colors[i%len(line_colors)]
        data_plot.plot(times+start, data[i]+gap*(n_plots-1-i), color=line_color)
        data_plot.plot(times+start, 0*data[i]+gap*(n_plots-1-i-0.5), color='k', linewidth=0.2)
        data_plot.text(start, gap*(n_plots-1-i), names[i], horizontalalignment='right', verticalalignment='center', color=line_color)
        if (peak_ind is not None) and show_detail:
            data_plot.plot(times[mark_inds[i]]+start, data[i][mark_inds[i]]+gap*(n_plots-1-i),'c.')
        if (threshold is not None) and show_detail:
            data_plot.plot(times+start, 0*data[i]+gap*(n_plots-1-i)+threshold[further_picks[i]]*1e6, 'g--', linewidth=0.5)
            data_plot.plot(times+start, 0*data[i]+gap*(n_plots-1-i)-threshold[further_picks[i]]*1e6, 'g--', linewidth=0.5)
        if (large_peak_ind is not None) and show_detail:
            data_plot.plot(times[large_mark_inds[i]]+start, data[i][large_mark_inds[i]]+gap*(n_plots-1-i),'r.')
        if (HFO_peak_ind is not None) or (HFO_time is not None):
            for j in range(len(HFO_mark_inds[i])):
                data_plot.plot(np.ones(10)*times[HFO_mark_inds[i][j]]+start, data[i][HFO_mark_inds[i][j]]+np.linspace(gap*(n_plots-1.3-i),gap*(n_plots-0.7-i),10),'m')
    data_plot.set_xlabel('time [s]')
    data_plot.text(start, -gap, 'spacing between channels: %.0f$\mu V$'%(gap), horizontalalignment='left', verticalalignment='center')
    plt.show()
    return data_plot

def plot_HFO_stat(raw, picks, HFO_time, HFO_amp, gap=1000):
    further_picks = []
    for i in range(len(picks)):
        if len(HFO_amp[i]) > 0:
            further_picks.append(i)
    further_picks = np.array(further_picks)
    names = np.array(raw.info['ch_names'])[picks[further_picks]]
    n_plots = len(further_picks)
    plt.figure(figsize=[14, 1.*n_plots])
    HFO_stat_plot = plt.gca()
    HFO_stat_plot.axis(ymin=-0.5*gap, ymax=(n_plots-0.5)*gap)
    HFO_stat_plot.set_yticks([])
    total_widths = []
    for i in range(n_plots):
        cursor_y = gap*(n_plots-1-i)
        cursor_x = 0
        color = 'b' if i%2 else 'r'
        for j in range(len(HFO_amp[further_picks[i]])):
            width = HFO_time[further_picks[i]][j][1]-HFO_time[further_picks[i]][j][0]
            height = 2*HFO_amp[further_picks[i]][j]
            this_rec = patches.Rectangle((cursor_x,cursor_y-0.5*height),width,height,fill=False,color=color)
            HFO_stat_plot.add_patch(this_rec)
            cursor_x += width
        HFO_stat_plot.text(0, gap*(n_plots-1-i), names[i], horizontalalignment='right', verticalalignment='center', color=color)
        total_widths.append(cursor_x)
    HFO_stat_plot.axis(xmax=1.25*np.max(np.array(total_widths)))
    HFO_stat_plot.set_xlabel('time [s]')
    HFO_stat_plot.text(0, -gap, 'spacing between channels: %.0f$\mu V$'%(gap), horizontalalignment='left', verticalalignment='center')
    #plt.show()
    return HFO_stat_plot

def plot_HFO_stat_absolute(raw, picks, HFO_time, HFO_amp, gap=1000):
    further_picks = []
    for i in range(len(picks)):
        if len(HFO_amp[i]) > 0:
            further_picks.append(i)
    further_picks = np.array(further_picks)
    names = np.array(raw.info['ch_names'])[picks[further_picks]]
    n_plots = len(further_picks)
    plt.figure(figsize=[14, 0.4 * n_plots])
    HFO_stat_plot = plt.gca()
    HFO_stat_plot.axis(ymin = -0.5 * gap, ymax = (n_plots - 0.5) * gap)
    HFO_stat_plot.set_yticks([])
    total_widths = []
    max_time_all = 0
    for i in range(n_plots):
        max_time = 0
        cursor_y = gap * (n_plots - 1 - i)
        cursor_x = 0
        color = 'b' if i%2 else 'r'
        for j in range(len(HFO_amp[further_picks[i]])):
            width = HFO_time[further_picks[i]][j][1] - HFO_time[further_picks[i]][j][0]
            height = 0.3 * HFO_amp[further_picks[i]][j]
            this_rec = patches.Rectangle((HFO_time[further_picks[i]][j][0], cursor_y - 0.5*height), width, height, fill = False, color = color)
            HFO_stat_plot.add_patch(this_rec)
            if HFO_time[further_picks[i]][j][1] > max_time:
                max_time = HFO_time[further_picks[i]][j][1]
            if HFO_time[further_picks[i]][j][1] > max_time_all:
                max_time_all = HFO_time[further_picks[i]][j][1]
            cursor_x += width
        line_x = np.linspace(0, max_time, 50)
        HFO_stat_plot.plot(line_x, cursor_y * np.ones(line_x.shape), '--', color = 'grey', linewidth = 0.3)
        HFO_stat_plot.text(0, gap * (n_plots - 1 - i), names[i], horizontalalignment = 'right', verticalalignment = 'center', color = color)
        total_widths.append(cursor_x)
    HFO_stat_plot.axis(xmin = 0, xmax = max_time_all)
    HFO_stat_plot.set_xlabel('time [s]')
    HFO_stat_plot.text(0, -2 * gap, 'spacing between channels: %.0f$\mu V$'%(gap), horizontalalignment = 'left', verticalalignment = 'center')
    #plt.show()
    return HFO_stat_plot

def HFO_result_concatenate(HFO_times_list, HFO_amps_list, picks_list, pick_names_list, raws_list):
    HFO_times_1 = HFO_times_list[0]
    HFO_amps_1 = HFO_amps_list[0]
    picks_1 = picks_list[0]
    pick_names_1 = pick_names_list[0]
    raws_11 = raws_list[0][0]
    raws_12 = raws_list[0][1]
    
    HFO_times_2 = HFO_times_list[1]
    HFO_amps_2 = HFO_amps_list[1]
    #picks_2 = picks_list[1]
    #pick_names_2 = pick_names_list[1]
    raws_21 = raws_list[1][0]
    raws_22 = raws_list[1][1]
    
    raws_11.append(raws_21)
    raws_12.append(raws_22)
    raws_concat = [raws_11, raws_12]
    picks_concat = picks_1
    pick_names_concat = pick_names_1
    
    for i in range(2):
        for j in range(len(HFO_times_1[0])):
            HFO_times_1[i][j] = HFO_times_1[i][j] + HFO_times_2[i][j]        
    HFO_times_concat = HFO_times_1
    
    for i in range(2):
        for j in range(len(HFO_times_1[0])):
            HFO_amps_1[i][j] = HFO_amps_1[i][j] + HFO_amps_2[i][j]        
    HFO_amps_concat = HFO_amps_1
    
    return HFO_times_concat, HFO_amps_concat, picks_concat, pick_names_concat, raws_concat

def find_HFO_long(raw, bands=[(80.,250.),(250.,500.)], abs_thresholds=[10.,5.], rel_thresholds=[2.,2.], picks='EEG', plot_names=None, start_time = 0):
    sfreq = raw.info["sfreq"]
    time_length = int(raw.get_data().shape[1] / sfreq)
    
    print('time_length = ' + str(time_length))
    
    TIME_LIMIT = 20
    crop_start = 0
    crop_end = crop_start + TIME_LIMIT
    while crop_end < time_length:
        raw_once = raw.copy().crop(crop_start, crop_end)
        print(str(crop_start) + ' to ' + str(crop_end))
        HFO_times, HFO_amps, picks, pick_names, raws = find_HFO(raw_once, bands = bands, abs_thresholds = abs_thresholds, rel_thresholds = rel_thresholds, picks = picks, plot_names=plot_names, start_time = start_time + crop_start)
        if crop_start == 0:
            HFO_times_pre = HFO_times
            HFO_amps_pre = HFO_amps
            picks_pre = picks
            pick_names_pre = pick_names
            raws_pre = raws
        else:
            HFO_times_list = [HFO_times_pre, HFO_times]
            HFO_amps_list = [HFO_amps_pre, HFO_amps]
            picks_list = [picks_pre, picks]
            pick_names_list = [pick_names_pre, pick_names]
            raws_list = [raws_pre, raws]
            HFO_times_pre, HFO_amps_pre, picks_pre, pick_names_pre, raws_pre = HFO_result_concatenate(HFO_times_list, HFO_amps_list, picks_list, pick_names_list, raws_list)
            
        crop_start += TIME_LIMIT 
        crop_end += TIME_LIMIT
        
    else:
        if time_length - crop_start > 5:  # make sure that the last section is longer than 5 seconds
            raw_once = raw.copy().crop(crop_start, time_length)
            print(str(crop_start) + ' to ' + str(time_length))
            HFO_times, HFO_amps, picks, pick_names, raws = find_HFO(raw_once, bands = bands, abs_thresholds = abs_thresholds, rel_thresholds = rel_thresholds, picks = picks, plot_names=plot_names, start_time = start_time + crop_start)
            
            try: 
                HFO_times_list = [HFO_times_pre, HFO_times]
                HFO_amps_list = [HFO_amps_pre, HFO_amps]
                picks_list = [picks_pre, picks]
                pick_names_list = [pick_names_pre, pick_names]
                raws_list = [raws_pre, raws]   
                HFO_times_pre, HFO_amps_pre, picks_pre, pick_names_pre, raws_pre = HFO_result_concatenate(HFO_times_list, HFO_amps_list, picks_list, pick_names_list, raws_list)
            except:
                HFO_times_pre = HFO_times
                HFO_amps_pre = HFO_amps
                picks_pre = picks
                pick_names_pre = pick_names
                raws_pre = raws
                
            
        else:
            pass
    
    return HFO_times_pre, HFO_amps_pre, picks_pre, pick_names_pre, raws_pre

def print_HFO_times(HFO_times, ch_names):
    for ch_name, HFO_time_list in zip(ch_names, HFO_times):
        print("Channel: " + ch_name)
        print(HFO_time_list)
    
if __name__ == "__main__":
    print("Welcome to the HFO analysis system (beta)!")
    mne.utils.logger.setLevel(mne.utils.logging.ERROR)
    
    DATA_LOADED = False
    data_path = input("Enter the full path of the .edf file:\n")
    
    #data_path = r"D:\Work\AIGE\waverecognition\data\wanghui_1.edf"#wanghui IID_EDF.edf"#shenshuaishuai_1.edf"
    while DATA_LOADED == False:
        try:
            raw = mne.io.read_raw_edf(data_path, preload=True)
        except Exception as e:
            data_path = input("Not a valid .edf file, please re-enter the full path of the .edf file:\n")
        else:
            DATA_LOADED = True
            
    start_time = float(input("Specify start time (< %.2fs):"%raw._times[-1]))
    end_time = float(input("Specify end time (< %.2fs):"%raw._times[-1]))
    raw.crop(start_time, end_time)
    
    print("Finished loading data, starting to analyze...")
    HFO_times, HFO_amps, picks, pick_names, raws = find_HFO_long(raw, start_time = start_time)
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    #input('Press any key to show the complete HFO analysis results for 80 to 250 Hz:\n')
    HFO_stat_plot = plot_HFO_stat_absolute(raw, picks, HFO_times[0], HFO_amps[0], gap=10)
    HFO_stat_plot.set_title('80-250Hz, absolute time')

    #input('Press any key to show the compl29.8ete HFO analysis results for 250 to 500 Hz:\n')    input('Press any key to show the complete HFO analysis results for 250 to 500 Hz:\n')

    HFO_stat_plot = plot_HFO_stat_absolute(raw, picks, HFO_times[1], HFO_amps[1], gap=10)
    HFO_stat_plot.set_title('250-500Hz, absolute time')

    #input('Press any key to show the complete HFO analysis results for 80 to 250 Hz:\n')    
    HFO_stat_plot = plot_HFO_stat(raw, picks, HFO_times[0], HFO_amps[0], gap=400)
    HFO_stat_plot.set_title('80-250Hz, stacked')

    #input('Press any key to show the complete HFO analysis results for 250 to 500 Hz:\n')    
    HFO_stat_plot = plot_HFO_stat(raw, picks, HFO_times[1], HFO_amps[1], gap=200)
    HFO_stat_plot.set_title('250-500Hz, stacked')
    
#    print_HFO_times(HFO_times[0], raw.info["ch_names"])
#     with open('Ripple_times.csv', 'w') as csvfile:
#         spamwriter = csv.writer(csvfile,dialect='excel')
#         for i in range(len(pick_names)):
#             spamwriter.writerow([pick_names[i]]+['%.2f-%.2fs'%(pair[0],pair[1]) for pair in HFO_times[0][i]])
#     with open('FastRipple_times.csv', 'w') as csvfile:
#         spamwriter = csv.writer(csvfile,dialect='excel')
#         for i in range(len(pick_names)):
#             spamwriter.writerow([pick_names[i]]+['%.2f-%.2fs'%(pair[0],pair[1]) for pair in HFO_times[1][i]])
# #        spamwriter.writerows(HFO_times[0])

#save ripple and fastripple dets, only chan with #HFO>0
    ripple_dets={}
    fastripple_dets={}
    for i in range(len(pick_names)):
        if len(HFO_times[0][i])>0:
            ripple_dets[pick_names[i]]=np.array(HFO_times[0][i])
        if len(HFO_times[1][i])>0:
            fastripple_dets[pick_names[i]]=np.array(HFO_times[1][i])
    with open('ripple_dets.pik','wb') as pikfile:
        pik.dump(ripple_dets,pikfile)
    with open('fastripple_dets.pik','wb') as pikfile:
        pik.dump(fastripple_dets,pikfile)




    # plt.show()
    # print r"plot_data(raw, picks, plot_names=[u'POL C\'2',u'POL C\'3',u'POL F\'6'], gap=2000, start=29.6, stop=29.8, HFO_time=HFO_times[0], line_colors=['c','r'])"
    
    
    
#    while(True):
#        x = input('if continue, press 1; else: press 0: ')
#        if x == '0':
#            break
#        else:
#            channel_name = unicode(input("Specific channel name: "))
#            start_time = float(input("Specify start time: "))
#            end_time = float(input("Specify end time: "))
#            plot_data(raw, picks, plot_names=[channel_name], gap=3000, start=start_time, stop=end_time)
            
    #input('Press Enter to exit...')