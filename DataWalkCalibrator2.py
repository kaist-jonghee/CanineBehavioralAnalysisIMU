from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, find_peaks
from tqdm import tqdm

Hz = 36 # Hz applied for all constraints

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def baseline_correction(signal, polynomial_order=3):
    """
    Apply baseline correction to a given signal using polynomial fitting.

    Args:
    signal (numpy.ndarray): The input signal to be baseline-corrected.
    polynomial_order (int): The order of the polynomial to fit to the baseline.

    Returns:
    numpy.ndarray: The baseline-corrected signal.
    """
    # Ensure signal is a numpy array
    signal = np.asarray(signal)

    # Fit a polynomial to the signal
    baseline = np.polyfit(np.arange(len(signal)), signal, polynomial_order)

    # Subtract the fitted baseline from the original signal
    baseline_corrected_signal = signal - np.polyval(baseline, np.arange(len(signal)))

    return baseline_corrected_signal

def filter(x):
    # Result of Plot 1
    # Original --> Final Filtered Signal
    signal = baseline_correction(x)

    filtered_signal = butter_bandpass_filter(signal,0.2,6,fs=Hz)
    fs = Hz
    cross_correlation = np.correlate(signal, filtered_signal, mode='full')
    delay = len(signal) - np.argmax(cross_correlation)
    phase_shift = delay / fs
    # Phase correction
    corrected_signal = np.roll(filtered_signal, -int(phase_shift * fs))

    def window_mean(x,window):
        pad = window//2
        padded = np.pad(x,(pad,pad),'edge')
        return np.array([np.mean(padded[i:i+window]) for i in range(len(x))])
    
    final_filtered_signal = corrected_signal - window_mean(corrected_signal,20)
    return final_filtered_signal

def filter_all(x):
    # 6xtimestamp
    for i in range(6):
        x[i,:]=filter(x[i,:])
    return x

def get_peaks(imu_filtered, distance_sec = 0.1, acc_prominence=0.1, gyr_prominence=20):
    filteraX = imu_filtered[0,:].T
    filteraY = imu_filtered[1,:].T
    filteraZ = imu_filtered[2,:].T
    filtergX = imu_filtered[3,:].T
    filtergY = imu_filtered[4,:].T
    filtergZ = imu_filtered[5,:].T
    
    peaksaX, _ = find_peaks(filteraX,distance=int(Hz*distance_sec),prominence=acc_prominence)
    peaksaY, _ = find_peaks(filteraY,distance=int(Hz*distance_sec),prominence=acc_prominence)
    peaksaZ, _ = find_peaks(filteraZ,distance=int(Hz*distance_sec),prominence=acc_prominence)
    peaksgX, _ = find_peaks(filtergX,distance=int(Hz*distance_sec),prominence=gyr_prominence)
    peaksgY, _ = find_peaks(filtergY,distance=int(Hz*distance_sec),prominence=gyr_prominence)
    peaksgZ, _ = find_peaks(filtergZ,distance=int(Hz*distance_sec),prominence=gyr_prominence)
    
    return peaksaX, peaksaY, peaksaZ, peaksgX, peaksgY, peaksgZ

def get_peak_present_amp(peakamps, total_len, interval = 0.5):
    # more than 5 peaks are needed for less than interval separated peaks! (5 peaks --> 4*0.5 --> 2초 이상 데이터!!)
    interval_len = interval*Hz
    if len(peakamps) == 0:
        return [], np.zeros((total_len,))
    peakamps = sorted(peakamps,key = lambda x: x[0])
    df = pd.DataFrame(peakamps)
    peakamps = df.groupby(0).max().reset_index().values
    # num * 2'
    # peaks = peakamps[:,0]

    selected = []
    temp = []
    if len(peakamps)>=2:
        prevp = peakamps[0][0]
        preva = peakamps[0][1]
        for [p,a] in peakamps[1:]:
            if p-prevp<=interval_len:
                temp.append((prevp,preva))
                temp.append((p,a))
            else:
                temp = sorted(list(set(temp)))
                if len(temp)>5:
                    selected.append(temp)
                temp = []
            prevp = p
            preva = a

        if len(temp)>5:
            selected.append(temp)

    amplitudes = np.zeros((total_len,))
    for s in selected:
        for i in range(len(s)-1):
            amplitudes[int(s[i][0]):int(s[i+1][0])+1] = max(s[i][1],s[i+1][1])# 의도적으로 계속 마지막거는 덮어씌워지도록 함!!

    selected = [[int(s[0][0]),int(s[-1][0])+1] for s in selected]

    return selected, amplitudes

def filter_binary_list(binary_list, window=1, density=0.8):
    filtered_list = []
    for i in range(len(binary_list)):
        filtered_list.append(binary_list[i])
        if binary_list[i] == 1:
            window_data = binary_list[i:min(len(binary_list),i+Hz*window)]
            window_density = sum(window_data)/len(window_data)
            if (window_density>=density) and (binary_list[min(len(binary_list),i+Hz*window)-1]==1):
                binary_list[i:min(len(binary_list),i+Hz*window)] = 1
        else:
            window_data = binary_list[i:min(len(binary_list),i+Hz*window)]
            window_density = sum(window_data)/len(window_data)
            if (window_density<=0.4) and (binary_list[min(len(binary_list),i+Hz*window)-1]==0):
                binary_list[i:min(len(binary_list),i+Hz*window)] = 0

    return filtered_list

def get_density_list(binary_list, window=1):
    filtered_list = []
    for i in range(len(binary_list)):
        window_data = binary_list[max(0,i-Hz*window):i+1]
        window_density = sum(window_data)/len(window_data)
        filtered_list.append(window_density)
    return filtered_list

def telepod_to_kookipod(tdata):
    '''
    input : 6 x timestamp (telepod data)
    output: 6 x timestamp (kookipod data)
    '''
    kdata = tdata
    kdata[0] = -tdata[1]
    kdata[1] = -tdata[0]
    kdata[3] = -tdata[4]
    kdata[4] = -tdata[3]
    return kdata

def plot_walk(input, timestamp):
    '''
    input : 6 x timestamp
    timestamp : (timestamp,)
    '''

    # input = telepod_to_kookipod(input)
    
    acc_min_amplitude = 0.1
    acc_max_amplitude = 2
    gyr_min_amplitude = 50
    gyr_max_amplitude = 400
    
    imu_filtered = filter_all(input)
    filteraX, filteraY, filteraZ, filtergX, filtergY, filtergZ = imu_filtered
    peaksaX, peaksaY, peaksaZ, peaksgX, peaksgY, peaksgZ = get_peaks(imu_filtered)

    paaX = np.stack([peaksaX,filteraX[peaksaX]]).T
    paaY = np.stack([peaksaY,filteraY[peaksaY]]).T
    paaZ = np.stack([peaksaZ,filteraZ[peaksaZ]]).T
    _, amplitudes_a = get_peak_present_amp(np.concatenate([paaX,paaY,paaZ]),input.shape[1])
    selected01_acc = amplitudes_a>0
    selected_thresh_acc = (amplitudes_a>acc_min_amplitude) & (amplitudes_a<=acc_max_amplitude)
    
    pagX = np.stack([peaksgX,filtergX[peaksgX]]).T
    pagY = np.stack([peaksgY,filtergY[peaksgY]]).T
    pagZ = np.stack([peaksgZ,filtergZ[peaksgZ]]).T
    _, amplitudes_g = get_peak_present_amp(np.concatenate([pagX,pagY,pagZ]),input.shape[1])
    selected01_gyr = amplitudes_g>0
    selected_thresh_gyr = (amplitudes_g>gyr_min_amplitude) & (amplitudes_g<=gyr_max_amplitude)

    binary = (selected01_acc & selected01_gyr) & (selected_thresh_acc | selected_thresh_gyr)
    filtered = filter_binary_list(binary)
    filtered_prob = get_density_list(binary)

    plt.figure(figsize=(10, 10))
    
    plt.subplot(6,1,1)
    plt.title('Acceleration')
    plt.plot(timestamp, imu_filtered[:3].T)
    plt.scatter(timestamp[peaksaX],filteraX[peaksaX],c='r')
    plt.scatter(timestamp[peaksaY],filteraY[peaksaY],c='r')
    plt.scatter(timestamp[peaksaZ],filteraZ[peaksaZ],c='r')

    plt.subplot(6,1,2)
    plt.title('Continuous Peaks (Acc)')
    plt.plot(timestamp,selected01_acc*max(amplitudes_a), label='continuous peaks')
    plt.plot(timestamp,selected_thresh_acc*max(amplitudes_a), label='continuous peaks + thresh')
    plt.plot(timestamp,amplitudes_a, label = 'amplitudes')
    plt.legend(loc='upper right')

    plt.subplot(6,1,3)
    plt.title('Gyroscope')
    plt.plot(timestamp, imu_filtered[3:].T)
    plt.scatter(timestamp[peaksgX],filtergX[peaksgX],c='r')
    plt.scatter(timestamp[peaksgY],filtergY[peaksgY],c='r')
    plt.scatter(timestamp[peaksgZ],filtergZ[peaksgZ],c='r')

    plt.subplot(6,1,4)
    plt.title('Continuous Peaks (Gyr)')
    plt.plot(timestamp,selected01_gyr*max(amplitudes_g), label='continuous peaks')
    plt.plot(timestamp,selected_thresh_gyr*max(amplitudes_g), label='continuous peaks + thresh')
    plt.plot(timestamp,amplitudes_g, label = 'amplitudes')
    plt.legend(loc='upper right')

    plt.subplot(6,1,5)
    plt.title('Continuous Peaks (Acc & Gyr)')
    plt.plot(timestamp,selected01_acc & selected01_gyr, label='continuous peaks')
    plt.legend(loc='upper right')

    plt.subplot(6,1,6)
    plt.title('Continuous Peaks + thresh (Acc & Gyr)')
    plt.plot(timestamp,binary, label='continuous peaks')
    plt.plot(timestamp,filtered, label='continuous peaks filtered')
    plt.plot(timestamp,filtered_prob,label='continuous peaks prob')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_pedometer(input, timestamp, precise = True):
    '''
    input : 6 x timestamp
    timestamp : (timestamp,)
    '''

    # input = telepod_to_kookipod(input)

    #region: for walkingincircles
    gY = input[-2,:]
    filteredgyr = signal.savgol_filter(gY,window_length=Hz,polyorder = 1)
    bandstopped = butter_bandstop_filter(filteredgyr,1,2/0.3,Hz)
    bandstopped_processed = bandstopped.copy()
    bandstopped_processed[np.abs(bandstopped_processed)<100]=0
    if sum(bandstopped_processed!=0):
        spinindex = np.where(bandstopped_processed!=0)[0]
        temp = list((spinindex[1:]-1)!=spinindex[:-1])
        startingindex = [True] + temp
        endingindex = temp + [True]

        startingindex = spinindex[startingindex]
        endingindex = spinindex[endingindex]
        endingindex += 1 # Making it [start,end)

        spiningtime = (endingindex-startingindex)/Hz
        selectedindex = np.where(spiningtime>=1.5)[0]

        startingindex = startingindex[selectedindex]
        endingindex = endingindex[selectedindex]
        degree = []
        state = []

        for (s,e) in zip(startingindex,endingindex):
            spindata = bandstopped[s:e]
            spindegree = round(sum(spindata)/Hz,2)
            degree.append(spindegree)
            if np.abs(spindegree)>=1000:
                if spindegree<0:
                    state.append("WalkingInCircles - CounterClockwise")
                else:
                    state.append("WalkingInCircles - Clockwise")
            else:
                if spindegree<0:
                    state.append("Curving - CounterClockwise")
                else:
                    state.append("Curving - Clockwise")
    else:
        startingindex=[]
        endingindex=[]
        degree = []
        state = []
    # if len(startingindex)>0:
    #     print(startingindex)
    #     # assert False
    #     print(endingindex)
    #     print(degree)
    #     print(state)
    # print("wallkingincircles done")
    # endregion: walkingincircles

    acc_min_amplitude = 0.1
    acc_max_amplitude = 2
    gyr_min_amplitude = 50
    gyr_max_amplitude = 400
    
    imu_filtered = filter_all(input)
    filteraX, filteraY, filteraZ, filtergX, filtergY, filtergZ = imu_filtered

    # print("filtering done")
    peaksaX, peaksaY, peaksaZ, peaksgX, peaksgY, peaksgZ = get_peaks(imu_filtered)
    # print("Peak finding done")

    paaX = np.stack([peaksaX,filteraX[peaksaX]]).T
    paaY = np.stack([peaksaY,filteraY[peaksaY]]).T
    paaZ = np.stack([peaksaZ,filteraZ[peaksaZ]]).T
    _, amplitudes_a = get_peak_present_amp(np.concatenate([paaX,paaY,paaZ]),input.shape[1])
    selected01_acc = amplitudes_a>0
    selected_thresh_acc = (amplitudes_a>acc_min_amplitude) & (amplitudes_a<=acc_max_amplitude)
    
    pagX = np.stack([peaksgX,filtergX[peaksgX]]).T
    pagY = np.stack([peaksgY,filtergY[peaksgY]]).T
    pagZ = np.stack([peaksgZ,filtergZ[peaksgZ]]).T
    _, amplitudes_g = get_peak_present_amp(np.concatenate([pagX,pagY,pagZ]),input.shape[1])
    selected01_gyr = amplitudes_g>0
    selected_thresh_gyr = (amplitudes_g>gyr_min_amplitude) & (amplitudes_g<=gyr_max_amplitude)

    rough_binary = selected01_acc & selected01_gyr
    binary = rough_binary & (selected_thresh_acc | selected_thresh_gyr)
    # print("Walk Detect Done")
    filtered = filter_binary_list(binary)
    # print("Walk Detect Done")
    filtered_prob = get_density_list(binary)
    # print("Walk Detect Done")


    # Running Detection
    a_idx = np.where(amplitudes_a>=1)[0]
    g_idx = np.where(amplitudes_g>=250)[0]

    def group_numbers(numbers):
        groups = []
        current_group = [numbers[0]]
    
        for num in numbers[1:]:
            if num - current_group[-1] <= Hz:
                current_group.append(num)
            else:
                if len(current_group) >= 3:
                    groups.append((current_group[0], current_group[-1]))
                current_group = [num]
    
        # Add the last group if it has at least 3 elements
        if len(current_group) >= 3:
            groups.append((current_group[0], current_group[-1]))
    
        return groups

    def perform_and_operation(groups1, groups2):
        result_groups = []
    
        for group1 in groups1:
            for group2 in groups2:
                start1, end1 = group1
                start2, end2 = group2
    
                # Check if there's an intersection between the two groups
                if start2 <= end1 + Hz and end2 >= start1 - Hz:
                    # Calculate the intersection range
                    intersection_start = max(start1, start2)
                    intersection_end = min(end1, end2)
    
                    # Check if the intersection range meets the criteria
                    if intersection_end - intersection_start >= Hz*2:
                        intersection_range = (intersection_start, intersection_end)
                        if intersection_range not in result_groups:
                            result_groups.append(intersection_range)
    
        return result_groups
    if len(a_idx)>0 and len(g_idx)>0:
        run_a = group_numbers(a_idx)
        run_g = group_numbers(g_idx)
        run = perform_and_operation(run_a,run_g)
    else:
        run = []
    
    # plt.subplot(2,1,1)
    # plt.plot(timestamp, amplitudes_a)
    # for (s,e) in run_a:
    #     plt.axvspan(timestamp[s],timestamp[e],alpha=0.3)
    # for (s,e) in run:
    #     plt.axvspan(timestamp[s],timestamp[e],color='red',alpha=0.3)
    # plt.subplot(2,1,2)
    # plt.plot(timestamp, amplitudes_g)
    # for (s,e) in run_g:
    #     plt.axvspan(timestamp[s],timestamp[e],alpha=0.3)
    # for (s,e) in run:
    #     plt.axvspan(timestamp[s],timestamp[e],color='red',alpha=0.3)
    # plt.show()
    # print("running done")
    # Running Detection Done
    
    # Pedometer
    peaX = np.zeros((len(timestamp,)))
    peaY = np.zeros((len(timestamp,)))
    peaZ = np.zeros((len(timestamp,)))
    pegX = np.zeros((len(timestamp,)))
    pegY = np.zeros((len(timestamp,)))
    pegZ = np.zeros((len(timestamp,)))

    peaX[peaksaX] = 1
    peaY[peaksaY] = 1
    peaZ[peaksaZ] = 1
    pegX[peaksgX] = 1
    pegY[peaksgY] = 1
    pegZ[peaksgZ] = 1

    # Filter!!
    # peaX = peaX & filtered
    # peaY = peaY & filtered
    # peaZ = peaZ & filtered
    # pegX = pegX & filtered
    # pegY = pegY & filtered
    # pegZ = pegZ & filtered

    def find_continuous_ones(binary_array):
        result = []
        start_index = None
        
        for i in range(len(binary_array)):
            if binary_array[i] == 1:
                if start_index is None:
                    start_index = i
            else:
                if start_index is not None:
                    result.append([start_index, i])
                    start_index = None
        
        # Check if the last sequence of ones continues till the end
        if start_index is not None:
            result.append([start_index, len(binary_array)])
        
        return result

    def space_btw_1s(binary_array):
        count = 0
        first = True
        out = []
        for i in binary_array:
            if i==1 and first:
                first = False
            elif i==1:
                out.append(count+1)
                count = 0
            else:
                count+=1
        return out

    if precise:
        rough_binary = filtered
        walk_binary = rough_binary
    
    se = find_continuous_ones(rough_binary) # [s,e] --> b[s:e] == 1
    out = []
    for [s,e] in se:
        daX = space_btw_1s(peaX[s:e])
        daY = space_btw_1s(peaY[s:e])
        daZ = space_btw_1s(peaZ[s:e])
        dgX = space_btw_1s(pegX[s:e])
        dgY = space_btw_1s(pegY[s:e])
        dgZ = space_btw_1s(pegZ[s:e])

        # print(len(daX),len(daY),len(daZ),len(dgX),len(dgY),len(dgZ))
        counts = [len(daX)+1,len(daY)+1,len(daZ)+1,len(dgX)+1,len(dgY)+1,len(dgZ)+1]
        dists = [np.mean(daX),np.mean(daY),np.mean(daZ),np.mean(dgX),np.mean(dgY),np.mean(dgZ)]
        conf = [np.std(daX)/np.mean(daX),np.std(daY)/np.mean(daY),np.std(daZ)/np.mean(daZ),np.std(dgX)/np.mean(dgX),np.std(dgY)/np.mean(dgY),np.std(dgZ)/np.mean(dgZ)]

        # validity check 1
        v1 = counts[-2]/counts[-1]
        v2 = counts[2]/counts[-1]
        v3 = dists[-1]/dists[-2]
        v4 = dists[-1]/dists[2]

        minth = 1.7
        maxth = 2.3

        gZ_validity_params = np.array([v1,v2,v3,v4])
        gZ_validity = (gZ_validity_params>minth) & (gZ_validity_params<maxth)
        gZ_valid = np.any(gZ_validity)

        # print(gZ_valid, v1,v2,v3,v4)
        # print(counts[2], counts[4], np.abs(counts[2]-counts[4]))

        # walk_info [step count (left,right each counted as one step), avg time taken for each step]
        selected_idx = None
        if gZ_valid and (conf[-1]<=0.4):
            walk_info = [counts[-1]*2-1, dists[-1]/2/Hz]
            selected_idx = 5
        elif counts[2] == counts[-2]:
            if conf[2] < conf[-2]:
                walk_info = [counts[2], dists[2]/Hz]
                selected_idx=2
            else:
                walk_info = [counts[-2], dists[-2]/Hz]
                selected_idx=4
        elif np.abs(counts[2]-counts[-2]) <= 2:
            # print(np.array(counts)==counts[2])
            # print(counts)
            dupaZ = sum(np.array(counts)==counts[2])
            dupgY = sum(np.array(counts)==counts[-2])
            if dupaZ>dupgY:
                walk_info = [counts[2], dists[2]/Hz]
                selected_idx=2
            elif dupaZ<dupgY:
                walk_info = [counts[-2], dists[-2]/Hz]
                selected_idx=4
            else:
                if conf[2] < conf[-2]:
                    walk_info = [counts[2], dists[2]/Hz]
                    selected_idx=2
                else:
                    walk_info = [counts[-2], dists[-2]/Hz]
                    selected_idx=4
        else:
            count = np.array(counts)
            cscores = []
            for i in range(5):
                cscores.append(np.absolute(count-count[i]))
            if np.min(cscores) <1.5:
                idx = np.argmin(cscores)
            else:
                idx = np.argmin(conf[:-1])
            selected_idx=idx
            walk_info = [counts[idx],dists[idx]/Hz]

        out.append([s,e,counts,dists,conf,selected_idx,walk_info])


    # Features
    walks = []
    pedometer = []
    pedo_peaks = []
    gZppeaks = []
    pedo_signal = np.zeros((len(timestamp),))

    for [s,e,counts,dists,conf,selected_idx,walk_info] in out:
        walks.append([timestamp[s],timestamp[e-1]])
        pedometer.append(walk_info[0])

        pedo_signal[s:e] = imu_filtered[selected_idx][s:e]/max(np.absolute(imu_filtered[selected_idx][s:e]))
        ppeaks = np.array([peaksaX,peaksaY,peaksaZ,peaksgX,peaksgY,peaksgZ][selected_idx])
        ppeaks = list(ppeaks[(s<=ppeaks) & (ppeaks<e)])
        if selected_idx == 5:
            gZppeaks += [int(ppeaks[i] + (ppeaks[i+1]-ppeaks[i])/2) for i in range(len(ppeaks)-1)]
        pedo_peaks += ppeaks
    
    pedometer_timestamps = timestamp[pedo_peaks+gZppeaks]

    curves = []
    curve_degrees = []
    for (s,e,d,st) in zip(startingindex, endingindex, degree, state):
        curves.append([timestamp[s],timestamp[e]])
        curve_degrees+=degree

    runs = []
    for (s,e) in run:
        runs.append([timestamp[s],timestamp[e]])
    
    
    
    if False:
        plt.figure(figsize=(10, 10))
        
        plt.subplot(6,1,1)
        plt.title(f'Acceleration {len(peaksaX)} {len(peaksaY)} {len(peaksaZ)}')
        plt.plot(timestamp, imu_filtered[:3].T)
        plt.scatter(timestamp[peaksaX],filteraX[peaksaX],c='r')
        plt.scatter(timestamp[peaksaY],filteraY[peaksaY],c='r')
        plt.scatter(timestamp[peaksaZ],filteraZ[peaksaZ],c='r')

        plt.subplot(6,1,2)
        plt.title('Continuous Peaks (Acc)')
        plt.plot(timestamp,selected01_acc*max(amplitudes_a), label='continuous peaks')
        plt.plot(timestamp,selected_thresh_acc*max(amplitudes_a), label='continuous peaks + thresh')
        plt.plot(timestamp,amplitudes_a, label = 'amplitudes')
        plt.legend(loc='upper right')

        plt.subplot(6,1,3)
        plt.title(f'Gyroscope {len(peaksgX)} {len(peaksgY)} {len(peaksgZ)}')
        plt.plot(timestamp, imu_filtered[3:].T)
        plt.scatter(timestamp[peaksgX],filtergX[peaksgX],c='r')
        plt.scatter(timestamp[peaksgY],filtergY[peaksgY],c='r')
        plt.scatter(timestamp[peaksgZ],filtergZ[peaksgZ],c='r')
        for (s,e,d,st) in zip(startingindex, endingindex, degree, state):
            plt.axvspan(timestamp[s],timestamp[e],alpha=0.3)
            plt.text(timestamp[int((e+s)/2)],0,'{}degrees\n{}'.format(str(d),st),bbox=dict(facecolor='yellow', alpha=0.5))

        plt.subplot(6,1,4)
        plt.title('Continuous Peaks (Gyr)')
        plt.plot(timestamp,selected01_gyr*max(amplitudes_g), label='continuous peaks')
        plt.plot(timestamp,selected_thresh_gyr*max(amplitudes_g), label='continuous peaks + thresh')
        plt.plot(timestamp,amplitudes_g, label = 'amplitudes')
        plt.legend(loc='upper right')

        plt.subplot(6,1,5)
        plt.title('Continuous Peaks (Acc & Gyr)')
        plt.plot(timestamp,selected01_acc & selected01_gyr, label='continuous peaks')
        plt.legend(loc='upper right')

        pedo_signal = np.zeros((len(timestamp),))
        pedo_peaks = []
        pedo_axis = []
        pedo_steps = 0
        gZppeaks = []
        for [s,e,counts,dists,conf,selected_idx,walk_info] in out:
            # plt.text(timestamp[int(np.mean([s,e]))],0.5,f"{str(counts)}\n{str(dists)}\n{str(conf)}",ha='center')
            plt.text(timestamp[int(np.mean([s,e]))],0.5,f"{['aX','aY','aZ','gX','gY','gZ'][selected_idx]} axis\n{str(walk_info[0])} steps\n{np.round(walk_info[1],2)} seconds",ha='center')
            pedo_signal[s:e] = imu_filtered[selected_idx][s:e]/max(np.absolute(imu_filtered[selected_idx][s:e]))
            ppeaks = np.array([peaksaX,peaksaY,peaksaZ,peaksgX,peaksgY,peaksgZ][selected_idx])
            ppeaks = list(ppeaks[(s<=ppeaks) & (ppeaks<e)])
            if selected_idx == 5:
                gZppeaks += [int(ppeaks[i] + (ppeaks[i+1]-ppeaks[i])/2) for i in range(len(ppeaks)-1)]
            pedo_peaks += ppeaks
            pedo_axis.append(selected_idx)
            pedo_steps += walk_info[0]

        plt.subplot(6,1,6)
        pedo_axis = list(set(pedo_axis))
        pedo_axis = np.array(['aX','aY','aZ','gX','gY','gZ'])[pedo_axis]
        plt.title(f'Pedo Signal {pedo_steps} steps, {pedo_axis} used')
        plt.plot(timestamp, pedo_signal)
        plt.scatter(timestamp[pedo_peaks],pedo_signal[pedo_peaks],c='r')
        if len(gZppeaks)>0:
            plt.scatter(timestamp[gZppeaks],pedo_signal[gZppeaks],c='b')
        for (s,e) in run:
            plt.axvspan(timestamp[s],timestamp[e],color='red',alpha=0.3)
            plt.text(timestamp[int((e+s)/2)],0,'RUNNING',bbox=dict(facecolor='yellow', alpha=0.5))
        for (s,e,d,st) in zip(startingindex, endingindex, degree, state):
            plt.axvspan(timestamp[s],timestamp[e],alpha=0.3)
            plt.text(timestamp[int((e+s)/2)],0,'{}degrees\n{}'.format(str(d),st),bbox=dict(facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        plt.show()

    return np.array(walks), np.array(pedometer), np.array(pedometer_timestamps), np.array(pedo_signal), np.array(curves), np.array(curve_degrees), np.array(runs)

def plot_pedometer2(input, timestamp, bigdf,BigRanges,smalldf,SmallRanges, precise = True):
    '''
    input : 6 x timestamp
    timestamp : (timestamp,)
    '''
    # 1. Convert Data to KookiPod Axis
    # input = telepod_to_kookipod(input)
    
    # 2. Detect Walking in Circles before filtering the data
    #region: for walkingincircles // outputs (startingindex, endingindex, degree, state) - [start,end) all lists have equal size, the size reflects the number of WiC detected.
    gY = input[-2,:]
    filteredgyr = signal.savgol_filter(gY,window_length=Hz,polyorder = 1)
    bandstopped = butter_bandstop_filter(filteredgyr,1,2/0.3,Hz)
    bandstopped_processed = bandstopped.copy()
    walkingincirclesthreshold = 20
    bandstopped_processed[np.abs(bandstopped_processed)<walkingincirclesthreshold]=0
    if sum(bandstopped_processed!=0):
        spinindex = np.where(bandstopped_processed!=0)[0]
        temp = list((spinindex[1:]-1)!=spinindex[:-1])
        startingindex = [True] + temp
        endingindex = temp + [True]

        startingindex = spinindex[startingindex]
        endingindex = spinindex[endingindex]
        endingindex += 1 # Making it [start,end)

        spiningtime = (endingindex-startingindex)/Hz
        selectedindex = np.where(spiningtime>=1.5)[0]

        startingindex = startingindex[selectedindex]
        endingindex = endingindex[selectedindex]
        degree = []
        state = []

        for (s,e) in zip(startingindex,endingindex):
            spindata = bandstopped[s:e]
            spindegree = round(sum(spindata)/Hz,2)
            degree.append(spindegree)
            if np.abs(spindegree)>=1000:
                if spindegree<0:
                    state.append("WalkingInCircles - CounterClockwise")
                else:
                    state.append("WalkingInCircles - Clockwise")
            else:
                if spindegree<0:
                    state.append("Curving - CounterClockwise")
                else:
                    state.append("Curving - Clockwise")
    else:
        startingindex=[]
        endingindex=[]
        degree = []
        state = []
    
    curves = []
    curve_degrees = []
    for (s,e,d,st) in zip(startingindex, endingindex, degree, state):
        curves.append([timestamp[s],timestamp[e-1]])
        curve_degrees+=degree
    # endregion: walkingincircles 

    # 3. Filter the input IMU data // outputs (imu_filtered, filteraX~filtergZ)
    #region: Filtering the input IMU data
    imu_filtered = filter_all(input)
    filteraX, filteraY, filteraZ, filtergX, filtergY, filtergZ = imu_filtered
    #endregion

    # 4. Find peaks for each axis data
    #region:  distance_sec = 0.1, acc_prominence=0.1, gyr_prominence=20 // outputs (peaksaX~peaksgZ)
    peaksaX, peaksaY, peaksaZ, peaksgX, peaksgY, peaksgZ = get_peaks(imu_filtered)
    npeaksaX, npeaksaY, npeaksaZ, npeaksgX, npeaksgY, npeaksgZ = get_peaks(-imu_filtered)
    #endregion

    # 5. Detect Jump / Shake
    # region: Jump Detection // outputs jumpout (indices of jump peaks) , shake_ranges [start,end)
    # findpeaks with aZ for jump
    # findpeaks with aY for shake
    # Jump if not consecutive, shake if consecutive.
    jumpaZ, _ = find_peaks(filteraZ,prominence=3)
    shakeaY, _ = find_peaks(filteraY,prominence=1.5)
    nshakeaY, _ = find_peaks(-filteraY,prominence=1.5)
    
    def filter_cont_jumps(numbers): # Removes Peaks that are connected with other peaks within 1 second
        # Filter outs continuous jumps (within 1seconds)
        if len(numbers)==0:
            return []
        groups = []
        current_group = [numbers[0]]
    
        for num in numbers[1:]:
            if num - current_group[-1] <= Hz:
                current_group.append(num)
            else:
                if len(current_group) >= 2: # 연결되어있는 것이 있으면 groups에 추가한다.
                    groups.append((current_group[0], current_group[-1]))
                current_group = [num]
        if len(current_group) >= 2:
            groups.append((current_group[0], current_group[-1]))
        
        if len(groups)>0:
            return sorted(list(set(numbers)-set(np.concatenate(groups))))
        else:
            return numbers

    def filter_shake(numbers): # Selects only Peaks that are connected with other peaks within 0.3 seconds. outputs [[s,e],[s,e]...] data[s:e] == shake
        # shakeaY 를 인풋으로 받아들임
        if len(numbers)==0:
            return []
        groups = []
        current_group = [numbers[0]]
    
        for num in numbers[1:]:
            if num - current_group[-1] <= 8:
                current_group.append(num)
            else:
                if len(current_group) >= 2:
                    groups.append((current_group[0], current_group[-1]))
                current_group = [num]
    
        # Add the last group if it has at least 3 elements
        if current_group[-1]-current_group[0] >= 1.2*Hz:
            groups.append((current_group[0], current_group[-1]))

        for i in range(len(groups)):
            groups[i] = [groups[i][0],groups[i][-1]+1]
        return np.array(groups)

    def filter_out_shake_from_jump(jump_peaks,shake_ranges): # Removes Jump Peaks if it's inside shake ranges' +- 1second range.
        # shake 하기 전 1초, 하기 후 1초 안에는 점프를 하지 않는다는 가정
        if len(shake_ranges)==0:
            return np.array(jump_peaks)
        new_peaks = []
        for peak in jump_peaks:
            exclude = False
            metric = shake_ranges-peak
            if 0 in metric:
                exclude=True
            elif True in ( ( (metric[:,0])-Hz<0) & ( (metric[:,-1]+Hz) >0) ):
                exclude=True
            if not exclude:
                new_peaks.append(peak)
        return np.array(new_peaks)
    
    def merge_ranges(list1, list2):
        list1 = list(list1)
        list2 = list(list2)
        # Combine the two lists
        combined_list = list1 + list2
        
        # Sort the combined list by the start values of the ranges
        combined_list.sort(key=lambda x: x[0])
        
        # Initialize the merged list
        merged_list = []
        
        # Iterate through the sorted list and merge ranges
        for range_start, range_end in combined_list:
            if not merged_list or merged_list[-1][1] < range_start - 1:
                # If the merged list is empty or the current range does not overlap with the last merged range
                merged_list.append((range_start, range_end))
            else:
                # There is an overlap or adjacency, so merge the ranges
                merged_list[-1] = (merged_list[-1][0], max(merged_list[-1][1], range_end))
        
        return np.array(merged_list)

    jump1 = filter_cont_jumps(jumpaZ)
    shakeout = filter_shake(shakeaY)
    nshakeout = filter_shake(nshakeaY)
    jumpout = filter_out_shake_from_jump(jump1,shakeout)
    jumpout = filter_out_shake_from_jump(jumpout,nshakeout)
    shake_ranges = merge_ranges(shakeout,nshakeout)
    shake_ranges = np.array([(s,e+1) for (s,e) in shake_ranges])

    # Needs to be still before and after shake (at least for +- 0.5 seconds)
    real_shake_ranges = []
    for (s,e) in shake_ranges:
        frontstd = np.std(imu_filtered[:,max(0,s-12):s],axis=1)
        backstd = np.std(imu_filtered[:,e:min(e+12,imu_filtered.shape[1])],axis=1)

        if (sum(frontstd[:3] <= 0.2) >=2) and (sum(backstd[:3] <= 0.2) >= 2) and (sum(frontstd[3:] <= 100) >=2) and (sum(backstd[3:] <= 100) >= 2):
            plt.subplot(2,1,1)
            plt.plot(imu_filtered[:3,max(0,s-12):min(e+12,imu_filtered.shape[1])].T)
            plt.subplot(2,1,2)
            plt.plot(imu_filtered[3:,max(0,s-12):min(e+12,imu_filtered.shape[1])].T)
            plt.show()
            real_shake_ranges.append((s,e))
        
    shake_ranges = real_shake_ranges
    
    shakeaY, _ = find_peaks(filtergZ,prominence=1000)
    nshakeaY, _ = find_peaks(-filtergZ,prominence=1000)
    # plt.subplot(2,1,1)
    # plt.plot(filteraY)
    # for i in shakeaY:
    #     plt.scatter(i, filteraY[i],c='r')
    # plt.subplot(2,1,2)
    # plt.plot(filtergZ)
    # for i in shakeaY:
    #     plt.scatter(i, filtergZ[i],c='r')
    # plt.show()
    more_shake_ranges = [[p-18, p+18] for p in shakeaY]
    more_shake_ranges += [[p-18, p+18] for p in nshakeaY]
    shake_ranges = merge_ranges(shake_ranges, more_shake_ranges)
    # endregion

    # 6. Detect Scratch
    # region: Scratch Detection // outputs scratch_ranges [start, end)
    # high density oscillation for all axes
    # peaks are connected within 0.5 seconds, at least 3 peaks (3 scratches) are required
    def scratch_density(peaks, length):
        sden = np.zeros((length,))
        window = int(1*Hz)
        def get_range(index, window, length):
            # Reflective window
            mini = index-window//2
            maxi = index+window//2
            ranges = []

            ranges.append([max(0,mini),min(length,maxi)])
            
            if mini<0:
                ranges.append([0, -(mini)])
            if maxi>length:
                ranges.append([length-(maxi-length),length])
            return ranges

        for i in range(length):
            ranges = get_range(i,window,length)
            count = 0
            for r in ranges:
                count += sum((peaks>=r[0]) & (peaks<r[1]))
            sden[i]=count
        return np.array(sden>=6,dtype=int)

    def group_scratch(numbers):
        groups = []
        if len(numbers)==0:
            return groups
        current_group = [numbers[0]]
    
        for num in numbers[1:]:
            if num - current_group[-1] <= 12:
                current_group.append(num)
            else:
                if len(current_group) >= 3:
                    groups.append((current_group[0], current_group[-1]))
                current_group = [num]
    
        # Add the last group if it has at least 3 elements
        if len(current_group) >= 3:
            groups.append((current_group[0], current_group[-1]))
    
        return groups
    
    scratchaX, _ = find_peaks(filteraX)
    scratchaY, _ = find_peaks(filteraY)
    scratchaZ, _ = find_peaks(filteraZ)
    scratchgX, _ = find_peaks(filtergX,prominence=20)
    scratchgY, _ = find_peaks(filtergY,prominence=20)
    scratchgZ, _ = find_peaks(filtergZ,prominence=20)
    metric = (scratch_density(scratchaX,len(timestamp)) + scratch_density(scratchaY,len(timestamp)) + scratch_density(scratchaZ,len(timestamp)) + scratch_density(scratchgX,len(timestamp)) + scratch_density(scratchgY,len(timestamp)) + scratch_density(scratchgZ,len(timestamp)))>=5
    numbers = np.where(metric==1)[0]
    scratch_ranges1 = group_scratch(numbers)

    scratchaX, _ = find_peaks(-filteraX)
    scratchaY, _ = find_peaks(-filteraY)
    scratchaZ, _ = find_peaks(-filteraZ)
    scratchgX, _ = find_peaks(-filtergX,prominence=20)
    scratchgY, _ = find_peaks(-filtergY,prominence=20)
    scratchgZ, _ = find_peaks(-filtergZ,prominence=20)
    metric = (scratch_density(scratchaX,len(timestamp)) + scratch_density(scratchaY,len(timestamp)) + scratch_density(scratchaZ,len(timestamp)) + scratch_density(scratchgX,len(timestamp)) + scratch_density(scratchgY,len(timestamp)) + scratch_density(scratchgZ,len(timestamp)))>=5
    numbers = np.where(metric==1)[0]
    scratch_ranges2 = group_scratch(numbers)

    scratch_ranges = merge_ranges(scratch_ranges1, scratch_ranges2)
    # endregion: End of Scratch Detection

    # 7-1. Detect Walking
    # region: Walk Detection // outputs rough_binary , binary (1 where it's detected to be walk 0 otherwise)
    acc_min_amplitude = 0.1
    acc_max_amplitude = 2
    gyr_min_amplitude = 50
    gyr_max_amplitude = 400

    # Postivie Peaks
    paaX = np.stack([peaksaX,filteraX[peaksaX]]).T
    paaY = np.stack([peaksaY,filteraY[peaksaY]]).T
    paaZ = np.stack([peaksaZ,filteraZ[peaksaZ]]).T
    _, amplitudes_a = get_peak_present_amp(np.concatenate([paaX,paaY,paaZ]),input.shape[1])
    selected01_acc = amplitudes_a>0
    selected_thresh_acc = (amplitudes_a>acc_min_amplitude) & (amplitudes_a<=acc_max_amplitude)
    selected_thresh_acc = np.array(filter_binary_list(selected_thresh_acc, density=0.5))

    pagX = np.stack([peaksgX,filtergX[peaksgX]]).T
    pagY = np.stack([peaksgY,filtergY[peaksgY]]).T
    pagZ = np.stack([peaksgZ,filtergZ[peaksgZ]]).T
    _, amplitudes_g = get_peak_present_amp(np.concatenate([pagX,pagY,pagZ]),input.shape[1])
    selected01_gyr = amplitudes_g>0
    selected_thresh_gyr = (amplitudes_g>gyr_min_amplitude) & (amplitudes_g<=gyr_max_amplitude)
    selected_thresh_gyr = np.array(filter_binary_list(selected_thresh_gyr, density=0.5))

    rough_binary1 = selected01_acc & selected01_gyr
    # binary1 = rough_binary1 & (selected_thresh_acc | selected_thresh_gyr)
    binary1 = rough_binary1 & (selected_thresh_acc & selected_thresh_gyr)

    # Negative Peaks
    paaX = np.stack([npeaksaX,-filteraX[npeaksaX]]).T
    paaY = np.stack([npeaksaY,-filteraY[npeaksaY]]).T
    paaZ = np.stack([npeaksaZ,-filteraZ[npeaksaZ]]).T
    _, namplitudes_a = get_peak_present_amp(np.concatenate([paaX,paaY,paaZ]),input.shape[1])
    selected01_acc = namplitudes_a>0
    selected_thresh_acc = (namplitudes_a>acc_min_amplitude) & (namplitudes_a<=acc_max_amplitude)
    selected_thresh_acc = np.array(filter_binary_list(selected_thresh_acc, density=0.5))
    
    
    pagX = np.stack([npeaksgX,-filtergX[npeaksgX]]).T
    pagY = np.stack([npeaksgY,-filtergY[npeaksgY]]).T
    pagZ = np.stack([npeaksgZ,-filtergZ[npeaksgZ]]).T
    _, namplitudes_g = get_peak_present_amp(np.concatenate([pagX,pagY,pagZ]),input.shape[1])
    selected01_gyr = namplitudes_g>0
    selected_thresh_gyr = (namplitudes_g>gyr_min_amplitude) & (namplitudes_g<=gyr_max_amplitude)
    selected_thresh_gyr = np.array(filter_binary_list(selected_thresh_gyr, density=0.5))
    
    rough_binary2 = selected01_acc & selected01_gyr
    # binary2 = rough_binary2 & (selected_thresh_acc | selected_thresh_gyr)
    binary2 = rough_binary2 & (selected_thresh_acc & selected_thresh_gyr)

    rough_binary = (rough_binary1 | rough_binary2)
    binary = (binary1 | binary2)

    # Exclude Scratch from rough_binary/binary
    for (s,e) in scratch_ranges:
        rough_binary[s:e]=0
        binary[s:e]=0

    # Exclude Shake from rough_binary/binary
    for (s,e) in shake_ranges:
        rough_binary[s:e]=0
        binary[s:e]=0
    # endregion

    # 7-2. Detect Running
    # region: Running Detection // outputs run [s,e)
    # Just using the amplitudes of both a and g for running detection
    def group_numbers(numbers):
        groups = []
        current_group = [numbers[0]]
    
        for num in numbers[1:]:
            if num - current_group[-1] <= Hz:
                current_group.append(num)
            else:
                if len(current_group) >= 3:
                    groups.append((current_group[0], current_group[-1]))
                current_group = [num]
    
        # Add the last group if it has at least 3 elements
        if len(current_group) >= 3:
            groups.append((current_group[0], current_group[-1]))
    
        return groups

    def perform_and_operation(groups1, groups2):
        result_groups = []
    
        for group1 in groups1:
            for group2 in groups2:
                start1, end1 = group1
                start2, end2 = group2
    
                # Check if there's an intersection between the two groups
                if start2 <= end1 + Hz and end2 >= start1 - Hz:
                    # Calculate the intersection range
                    intersection_start = max(start1, start2)
                    intersection_end = min(end1, end2)
    
                    # Check if the intersection range meets the criteria
                    if intersection_end - intersection_start >= Hz*2:
                        intersection_range = (intersection_start, intersection_end)
                        if intersection_range not in result_groups:
                            result_groups.append(intersection_range)
    
        return result_groups
    
    a_idx = np.where(amplitudes_a>=1)[0]
    g_idx = np.where(amplitudes_g>=250)[0]
    if len(a_idx)>0 and len(g_idx)>0:
        run_a = group_numbers(a_idx)
        run_g = group_numbers(g_idx)
        run1 = perform_and_operation(run_a,run_g)
    else:
        run1 = []

    a_idx = np.where(namplitudes_a>=1)[0]
    g_idx = np.where(namplitudes_g>=250)[0]
    if len(a_idx)>0 and len(g_idx)>0:
        run_a = group_numbers(a_idx)
        run_g = group_numbers(g_idx)
        run2 = perform_and_operation(run_a,run_g)
    else:
        run2 = []

    run = merge_ranges(run1,run2)
    # endregion: Running Detection Done
    
    # 8. Change Walk to Pedometer (1 for datapoints that are detected as walk(a step))
    # region: Pedometer (set precise==True to use binary instead of rough_binary) // outputs (pedo_peaks, gZppeaks, pedo_axis, pedo_steps) pedo_peaks : pedometer peak indices, gZppeaks : Predicted pedometer peak indicies, pedo_steps : pedometer steps, pedo_axis : pedometer used axes
    peaX = np.zeros((len(timestamp,)))
    peaY = np.zeros((len(timestamp,)))
    peaZ = np.zeros((len(timestamp,)))
    pegX = np.zeros((len(timestamp,)))
    pegY = np.zeros((len(timestamp,)))
    pegZ = np.zeros((len(timestamp,)))
    peaX[peaksaX] = 1
    peaY[peaksaY] = 1
    peaZ[peaksaZ] = 1
    pegX[peaksgX] = 1
    pegY[peaksgY] = 1
    pegZ[peaksgZ] = 1

    def find_continuous_ones(binary_array):
        result = []
        start_index = None
        
        for i in range(len(binary_array)):
            if binary_array[i] == 1:
                if start_index is None:
                    start_index = i
            else:
                if start_index is not None:
                    result.append([start_index, i])
                    start_index = None
        
        # Check if the last sequence of ones continues till the end
        if start_index is not None:
            result.append([start_index, len(binary_array)])
        
        return result

    def space_btw_1s(binary_array):
        count = 0
        first = True
        out = []
        for i in binary_array:
            if i==1 and first:
                first = False
            elif i==1:
                out.append(count+1)
                count = 0
            else:
                count+=1
        return out

    if precise:
        rough_binary = binary
    
    se = find_continuous_ones(rough_binary) # [s,e] --> b[s:e] == 1
    out = []

    def safe_mean(array):
        if len(array) > 0:
            return np.mean(array)
        else:
            return np.inf
    
    def safe_std(array):
        if len(array) > 0:
            return np.std(array)
        else:
            return np.inf
    
    for [s,e] in se:
        daX = space_btw_1s(peaX[s:e])
        daY = space_btw_1s(peaY[s:e])
        daZ = space_btw_1s(peaZ[s:e])
        dgX = space_btw_1s(pegX[s:e])
        dgY = space_btw_1s(pegY[s:e])
        dgZ = space_btw_1s(pegZ[s:e])

        if sum([len(daX),len(daY),len(daZ),len(dgX),len(dgY),len(dgZ)]) == 0:
            continue

        counts = [len(daX)+1,len(daY)+1,len(daZ)+1,len(dgX)+1,len(dgY)+1,len(dgZ)+1]
        dists = [safe_mean(daX),safe_mean(daY),safe_mean(daZ),safe_mean(dgX),safe_mean(dgY),safe_mean(dgZ)]
        conf = [safe_std(daX)/safe_mean(daX),safe_std(daY)/safe_mean(daY),safe_std(daZ)/safe_mean(daZ),safe_std(dgX)/safe_mean(dgX),safe_std(dgY)/safe_mean(dgY),safe_std(dgZ)/safe_mean(dgZ)]
        conf = np.nan_to_num(conf,nan=np.inf)
        conf=np.array([100]*6)

        # validity check 1
        v1 = counts[-2]/counts[-1]
        v2 = counts[2]/counts[-1]
        v3 = dists[-1]/dists[-2]
        v4 = dists[-1]/dists[2]

        minth = 1.7
        maxth = 2.3

        gZ_validity_params = np.array([v1,v2,v3,v4])
        gZ_validity = (gZ_validity_params>minth) & (gZ_validity_params<maxth)
        gZ_valid = np.any(gZ_validity)

        # walk_info [step count (left,right each counted as one step), avg time taken for each step]
        selected_idx = None
        if gZ_valid and (conf[-1]<=0.4):
            walk_info = [counts[-1]*2-1, dists[-1]/2/Hz]
            selected_idx = 5
        elif counts[2] == counts[-2]:
            if conf[2] < conf[-2]:
                walk_info = [counts[2], dists[2]/Hz]
                selected_idx=2
            else:
                walk_info = [counts[-2], dists[-2]/Hz]
                selected_idx=4
        elif np.abs(counts[2]-counts[-2]) <= 2:
            dupaZ = sum(np.array(counts)==counts[2])
            dupgY = sum(np.array(counts)==counts[-2])
            if dupaZ>dupgY:
                walk_info = [counts[2], dists[2]/Hz]
                selected_idx=2
            elif dupaZ<dupgY:
                walk_info = [counts[-2], dists[-2]/Hz]
                selected_idx=4
            else:
                if conf[2] < conf[-2]:
                    walk_info = [counts[2], dists[2]/Hz]
                    selected_idx=2
                else:
                    walk_info = [counts[-2], dists[-2]/Hz]
                    selected_idx=4
        else:
            count = np.array(counts)
            cscores = []
            for i in range(5):
                cscores.append(np.mean(np.absolute(count-count[i])))
            if np.min(cscores) <1.5:
                idx = np.argmin(cscores)
            else:
                idx = np.argmin(conf[:-1])
            selected_idx=idx
            walk_info = [counts[idx],dists[idx]/Hz]

        out.append([s,e,counts,dists,conf,selected_idx,walk_info])

    pedo_signal = np.zeros((len(timestamp),))
    pedo_peaks = []
    pedo_axis = []
    pedo_steps = 0
    gZppeaks = []
    for [s,e,counts,dists,conf,selected_idx,walk_info] in out:
        pedo_signal[s:e] = imu_filtered[selected_idx][s:e]/max(np.absolute(imu_filtered[selected_idx][s:e]))
        ppeaks = np.array([peaksaX,peaksaY,peaksaZ,peaksgX,peaksgY,peaksgZ][selected_idx])
        ppeaks = list(ppeaks[(s<=ppeaks) & (ppeaks<e)])
        if selected_idx == 5:
            gZppeaks += [int(ppeaks[i] + (ppeaks[i+1]-ppeaks[i])/2) for i in range(len(ppeaks)-1)]
        pedo_peaks += ppeaks
        pedo_axis.append(selected_idx)
        pedo_steps += walk_info[0]
    pedo_axis = list(set(pedo_axis))
    pedo_axis = np.array(['aX','aY','aZ','gX','gY','gZ'])[pedo_axis]
    # endregion
    
    # 9. Pedometer Detected with ShapeDetector
    # region: process shapedetector results // outputs (pedometer, pedometer_time) pedometer: step count per walk chunk, pedometer_time: Step count times // outputs (smallsignal, small_medians) smallsignal : signal detected by shapedetector, small_medians : peaks detected by shapedetector
    def find_median_indices(lst):
        intervals = []
        start_index = None

        # Identify intervals bounded by 1 and -1
        for i, val in enumerate(lst):
            if val == 1:
                start_index = i
            elif val == -1 and start_index is not None:
                intervals.append((start_index, i))
                start_index = None

        # Calculate the median index for each interval
        median_indices = []
        for start, end in intervals:
            median_index = (start + end) // 2
            median_indices.append(median_index)

        return median_indices

    smallsignal = np.zeros((len(smalldf),))
    smallsignal[rough_binary] = smalldf.to_numpy().flatten()[rough_binary]
    small_medians = np.array(find_median_indices(smallsignal))
    pedometer = []
    pedometer_time = []
    for (s,e) in se:
        pedometer.append(sum( (small_medians>=s) & (small_medians<=e) ))
        for i in small_medians[( (small_medians>=s) & (small_medians<=e) )]:
            pedometer_time.append(timestamp[i])
    # endregion

    # 10. Shake/Scratch Detected with ShapeDetector
    # region: process shapedetector results // outputs (bigsignal, big_medians) bigsignal : signal detected by shapedetector, big_medians : peaks detected by shapedetector
    bigsignal = np.zeros((len(bigdf),))
    bigsignal[rough_binary] = bigdf.to_numpy().flatten()[rough_binary]
    big_medians = np.array(find_median_indices(bigsignal))
    # endregion

    # 0: Stationary, 1: walk, 2: run, 3: jump, 4: shake, 5: scratch
    rough_binary = rough_binary.astype(int)
    for (s,e) in run:
        iswalking = rough_binary[s:e].copy()
        # np.ones(iswalking.shape, dtype=int)*2*iswalking
        rough_binary[s:e] = np.ones(iswalking.shape, dtype=int)*2*iswalking
    for jump in jumpout:
        rough_binary[jump] = 3
    for (s,e) in shake_ranges:
        rough_binary[s:e] = 4
    for (s,e) in scratch_ranges:
        rough_binary[s:e] = 5

    return {'detections': np.array(rough_binary), 'walk_pedometer': np.array(pedometer), 'pedometer_time': np.array(pedometer_time), 'curve_startend':np.array(curves), 'curve_degree':np.array(curve_degrees)}
# region: Constraints


def BigWaveConstraint(input,timestamp,  prevranges=[], axis=['aX','aY','aZ','gX','gY','gZ'], acc_ampthr=1, acc_plenmin=0, acc_plenmax=0.3, acc_changemin=0, acc_changemax=0.3, gyr_ampthr=500, gyr_plenmin=0, gyr_plenmax=0.3, gyr_changemin=0, gyr_changemax=0.3,   duration=1, mode='c', verbose=True):
    # input = telepod_to_kookipod(input)
    df = pd.DataFrame(columns=['aX','aY','aZ','gX','gY','gZ'],data=input.T)
    df['time'] = timestamp
    
    # Checking previous constraint results & looking up for only those regions
    if len(prevranges)>0:
        if verbose:
            print("Applying the given prevranges")
        dfs2lookup = [df[(df['time']>=s) & (df['time']<=e)] for [s,e] in tqdm(prevranges)]
    else:
        dfs2lookup = [df]
    # Data structure to contain the time ranges that have passed the constraint
    SelectedRanges = []

    # TODO: Implement the constraint's logic
    if verbose:
        print("Applying BigWaveConstraint")

    def status_report(df,isacc=True):
        change = sum(sum((df!=0).to_numpy()))
        size = df.shape[0]*df.shape[1]
        
        if size==0:
            return

        if verbose:
            if isacc:
                # print(f"\t{change}/{size} zeroed out for ACC")
                print(f"\t{change}/{size} remain for ACC")
            else:
                # print(f"\t{change}/{size} zeroed out for GYR")
                print(f"\t{change}/{size} remain for GYR")

    def filter_continuous_segments(series, threshold_minlength, threshold_maxlength):
        n = len(series)
        result = pd.Series([0] * n, index=series.index)  # Initialize the result Series with zeros
        
        i = 0
        while i < n:
            if series.iloc[i] == 0:
                i += 1
                continue
            
            start = i
            value = series.iloc[i]
            
            # Find the end of the current segment
            while i < n and series.iloc[i] == value:
                i += 1
            end = i
            
            segment_length = end - start
            
            # Check if the segment length is within the specified thresholds
            if threshold_minlength <= segment_length <= threshold_maxlength:
                result.iloc[start:end] = value
        
        return result

    def zero_out_isolated_chunks(s, min_length, max_length):
        # Find the positions of 1s and -1s
        positions = s[(s == 1) | (s == -1)].index
        
        # Function to identify chunks
        def identify_chunks(positions):
            if len(positions) == 0:
                return []
            
            chunks = []
            current_chunk = [positions[0]]
            
            for pos in positions[1:]:
                if pos - current_chunk[-1] - 1 <= max_length:
                    current_chunk.append(pos)
                else:
                    chunks.append(current_chunk)
                    current_chunk = [pos]
            
            # Append the last chunk
            chunks.append(current_chunk)
            
            return chunks
        
        # Function to validate chunks
        def validate_chunks(chunks):
            valid_chunks = []
            for chunk in chunks:
                sub_s = s.loc[chunk]
                ones = sub_s[sub_s == 1].index
                minus_ones = sub_s[sub_s == -1].index
                
                # Check if chunk contains both 1s and -1s
                if not ones.empty and not minus_ones.empty:
                    # Check if sequences of 1s and -1s are separated by at least min_length
                    is_valid = True
                    for i in range(1, len(chunk)):
                        if s[chunk[i]] != s[chunk[i-1]] and (chunk[i] - chunk[i-1] - 1) < min_length:
                            is_valid = False
                            break
                    if is_valid:
                        valid_chunks.append(chunk)
            
            return valid_chunks
        
        # Identify chunks
        chunks = identify_chunks(positions)
        
        # Validate chunks
        valid_chunks = validate_chunks(chunks)
        
        # Identify positions in valid chunks
        valid_positions = [pos for chunk in valid_chunks for pos in chunk]
        
        # Zero out invalid positions
        invalid_positions = list(set(positions) - set(valid_positions))
        s.loc[invalid_positions] = 0
        
        return s

    def zero_out_non_alternating_groups(series, group_margin, life=3):
        n = len(series)
        i = 0
        groups = []

        # Find the start and end indices of each group
        while i < n:
            # Find the start of the next group
            while i < n and series.iloc[i] == 0:
                i += 1
            if i >= n:
                break
            start = i

            # Find the end of the current group
            zero_count = 0
            while i < n and zero_count < group_margin:
                if series.iloc[i] == 0:
                    zero_count += 1
                else:
                    zero_count = 0
                i += 1

            # Adjust the end to exclude trailing zeros counted for group_margin
            end = i-group_margin if zero_count >= group_margin else i
            groups.append((start, end))

        # Check each group for alternating chunks of 1s and -1s
        for start, end in groups:
            if start >= end:
                continue

            group = series.iloc[start:end]
            is_alternating = True
            time2see = 0
            prev = 0
            nummisses = 0

            for value in group:
                
                if value == 0: # 신경 안써도 됨
                    prev=value
                    continue
                elif time2see == 0: # 처음
                    time2see = -value
                    prev = value
                elif value==prev: # 같은 청크안에 있음
                    prev = value
                elif value==time2see: # 바뀜
                    time2see = -value
                    nummisses = 0
                    prev=value
                elif value!=time2see and nummisses<=life:
                    nummisses += 1
                    prev=value
                else:
                    is_alternating = False
                    break

            # Zero out the group if not alternating
            if not is_alternating:
                series.iloc[start:end] = 0

        return series

    def zero_out_small_groups(series, group_margin, group_length):
        n = len(series)
        i = 0
        groups = []

        # Find the start and end indices of each group
        while i < n:
            # Find the start of the next group
            while i < n and series.iloc[i] == 0:
                i += 1
            if i >= n:
                break
            start = i

            # Find the end of the current group
            zero_count = 0
            while i < n and zero_count < group_margin:
                if series.iloc[i] == 0:
                    zero_count += 1
                else:
                    zero_count = 0
                i += 1

            # Adjust the end to exclude trailing zeros counted for group_margin
            end = i-group_margin if zero_count >= group_margin else i
            groups.append((start, end))
        
        for start, end in groups:
            if end-start < group_length:
                series.iloc[start:end] = 0
        
        return series

    def mark_groups(series, group_margin):
        n = len(series)
        i = 0
        groups = []

        # Find the start and end indices of each group
        while i < n:
            # Find the start of the next group
            while i < n and series.iloc[i] == 0:
                i += 1
            if i >= n:
                break
            start = i

            # Find the end of the current group
            zero_count = 0
            while i < n and zero_count < group_margin:
                if series.iloc[i] == 0:
                    zero_count += 1
                else:
                    zero_count = 0
                i += 1

            # Adjust the end to exclude trailing zeros counted for group_margin
            end = i-group_margin if zero_count >= group_margin else i
            groups.append((start, end))
        
        check = np.zeros((n,))
        for (s,e) in groups:
            check[s:e]=1
        
        return check

    def find_continuous_true_indices(binary_list):
        start_index = None
        end_index = None
        continuous_ranges = []

        for i, value in enumerate(binary_list):
            if value:
                if start_index is None:
                    start_index = i
                end_index = i
            else:
                if start_index is not None:
                    continuous_ranges.append((start_index, end_index))
                    start_index = None
                    end_index = None

        # Check if there is a continuous range at the end
        if start_index is not None:
            continuous_ranges.append((start_index, end_index))

        return continuous_ranges

    for df in dfs2lookup:
        # temp = df[['aX','aY','aZ','gX','gY','gZ']].copy()
        # df[['aX','aY','aZ','gX','gY','gZ']] = telepod_to_kookipod(temp.to_numpy())

        # acc
        accaxis = [a for a in axis if a.startswith('a')]
        accdf = df[accaxis].copy()
        for axis in accaxis:
            accdf[axis] = filter(accdf[axis])
        # gyr
        gyraxis = [g for g in axis if g.startswith('g')]
        gyrdf = df[gyraxis].copy()
        for axis in gyraxis:
            gyrdf[axis] = filter(gyrdf[axis])

        # Zero out regions that don't fufill the thresholds
        if verbose:
            print("BigWaveConstraint-1 (Needs to Cross the Threshold)")
        if mode=='c':
            if len(accaxis)>0:
                if (len(accdf) in (accdf.abs()<acc_ampthr).sum(axis=0).to_numpy()):
                    continue
                accdf[accdf.abs() < acc_ampthr] = 0
                
            if len(gyraxis)>0:
                if (len(gyrdf) in (gyrdf.abs()<gyr_ampthr).sum(axis=0).to_numpy()):
                    continue
                gyrdf[gyrdf.abs() < gyr_ampthr] = 0
        else:
            assert False, f"{mode} mode hasn't been implemented"
        status_report(accdf)
        status_report(gyrdf,False)

        # Positive to 1, Negative to -1
        accdf = np.sign(accdf)
        gyrdf = np.sign(gyrdf)

        # Find regions with
        # 1. Alternating 1, -1
        # 2. Continuosly for longer than duration seconds
        # 3. distances between the 1 and -1 needs to be in chagemin~changemax sec (more than 3Hz)
        # 4. Each 1 and -1 chunks need to be in plenmin~plenmax seconds (INCLUSIVE)

        # number 4
        # If there are continuous 1s or -1s longer than plenmax seconds, zero them out.
        if verbose:
            print("BigWaveConstraint-2 (Needs to fall in the desired regions)")
        acc_threshold_minlength = acc_plenmin * Hz
        acc_threshold_maxlength = acc_plenmax * Hz
        for aa in accaxis:
            out = filter_continuous_segments(accdf[aa],acc_threshold_minlength,acc_threshold_maxlength)
            accdf[aa] = out
            if sum(out) == 0:
                continue
        gyr_threshold_minlength = gyr_plenmin * Hz
        gyr_threshold_maxlength = gyr_plenmax * Hz
        for ga in gyraxis:
            out = filter_continuous_segments(gyrdf[ga],gyr_threshold_minlength,gyr_threshold_maxlength)
            gyrdf[ga] = out
            if sum(out) == 0:
                continue
        status_report(accdf)
        status_report(gyrdf,False)

        # number 3
        # The distance between chunks need to be in chagemin~changemax sec.(INCLUSIVE) Zero other cases out
        if verbose:
            print("BigWaveConstraint-3 (Needs to match the cycle time)")
        acc_threshold_minlength = acc_changemin * Hz
        acc_threshold_maxlength = acc_changemax * Hz
        for aa in accaxis:
            out = zero_out_isolated_chunks(accdf[aa],acc_threshold_minlength,acc_threshold_maxlength)
            accdf[aa] = out
            if sum(out) == 0:
                continue
        gyr_threshold_minlength = gyr_changemin * Hz
        gyr_threshold_maxlength = gyr_changemax * Hz
        for ga in gyraxis:
            out = zero_out_isolated_chunks(gyrdf[ga],gyr_threshold_minlength,gyr_threshold_maxlength)
            gyrdf[ga] = out
            if sum(out) == 0:
                continue
        status_report(accdf)
        status_report(gyrdf,False)

        # number 2
        # Setting Group Size Minimum Bar
        if verbose:
            print("BigWaveConstraint-4 (Needs to be at least duration long)")
        acc_group_margin = int(acc_changemax * Hz)
        threshold_length = duration * Hz
        for aa in accaxis:
            out = zero_out_small_groups(accdf[aa],acc_group_margin,threshold_length)
            accdf[aa] = out
            if sum(out) == 0:
                continue
        gyr_group_margin = int(gyr_changemax * Hz)
        for ga in gyraxis:
            out = zero_out_small_groups(gyrdf[ga],gyr_group_margin,threshold_length)
            gyrdf[ga] = out
            if sum(out) == 0:
                continue
        status_report(accdf)
        status_report(gyrdf,False)
        
        # number 1
        if verbose:
            print("BigWaveConstraint-5 (The signal needs to be alternating. (lifes==3))")
        acc_group_margin = int(acc_changemax * Hz)
        for aa in accaxis:
            out = zero_out_non_alternating_groups(accdf[aa],acc_group_margin)
            accdf[aa] = out
            if sum(out) == 0:
                continue
        gyr_group_margin = int(gyr_changemax * Hz)
        for ga in gyraxis:
            out = zero_out_non_alternating_groups(gyrdf[ga],gyr_group_margin)
            gyrdf[ga] = out
            if sum(out) == 0:
                continue
        status_report(accdf)
        status_report(gyrdf,False)
        
        ranges = []
        acc_group_margin = int(acc_changemax * Hz)
        for aa in accaxis:
            ranges.append(mark_groups(accdf[aa],acc_group_margin))
        gyr_group_margin = int(gyr_changemax * Hz)
        for ga in gyraxis:
            ranges.append(mark_groups(gyrdf[ga],gyr_group_margin))

        result = [all(bit_lists) for bit_lists in zip(*ranges)]
        SelectedIndices = find_continuous_true_indices(result)
        timestamp = df['time'].to_numpy()
        SelectedRanges += [[timestamp[s],timestamp[e]] for [s,e] in SelectedIndices]
        

    # End of TODO
    num_passed = len(SelectedRanges)
    if verbose:
        print(f'A total of {num_passed} signals have passed the BigwaveConstraint')
    # Return the time ranges that have passed the constraint
    return SelectedRanges, accdf, gyrdf

# endregion

def run_plot_pedometer2(data, times):
    SmallRanges, smalldf, _ = BigWaveConstraint(data,times,  prevranges=[], axis=['aZ'], acc_ampthr=0.01, acc_plenmin=0, acc_plenmax=0.875, acc_changemin=0, acc_changemax=0.5, gyr_ampthr=20, gyr_plenmin=0, gyr_plenmax=0.5, gyr_changemin=0.1, gyr_changemax=0.3, duration=2, mode='c', verbose=False)
    BigRanges, _, bigdf = BigWaveConstraint(data,times,  prevranges=[], axis=['gX'], acc_ampthr=0.01, acc_plenmin=0, acc_plenmax=0.875, acc_changemin=0, acc_changemax=0.5, gyr_ampthr=200, gyr_plenmin=0, gyr_plenmax=2.0, gyr_changemin=0, gyr_changemax=2.0, duration=1, mode='c', verbose=False)    
    output = plot_pedometer2(data,times,bigdf,BigRanges,smalldf,SmallRanges)
    return output



def plot_output(data, output):
    """
    Visualizes accelerometer and gyroscope data with color-coded background based on detections.
    Adds a legend for activity labels.
    """
    detections = np.array(output['detections'])  # Ensure it's a NumPy array
    t = np.arange(len(detections))

    label_colors = {
        0: 'lightgray',  # Stationary
        1: 'lightgreen', # Walk
        2: 'orange',     # Run
        3: 'skyblue',    # Jump
        4: 'violet',     # Shake
        5: 'salmon'      # Scratch
    }

    label_names = {
        0: 'Stationary',
        1: 'Walk',
        2: 'Run',
        3: 'Jump',
        4: 'Shake',
        5: 'Scratch'
    }

    color_array = [label_colors.get(label, 'white') for label in detections]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot accelerometer
    axes[0].set_title('Accelerometer (aX, aY, aZ)')
    for i in range(3):
        axes[0].plot(data[i, :], label=f'a{i}')
    axes[0].legend(loc='upper right')

    # Plot gyroscope
    axes[1].set_title('Gyroscope (gX, gY, gZ)')
    for i in range(3, 6):
        axes[1].plot(data[i, :], label=f'g{i-3}')
    axes[1].legend(loc='upper right')

    # Add background color to both subplots
    for ax in axes:
        for i in range(len(detections)):
            ax.axvspan(i - 0.5, i + 0.5, color=color_array[i], alpha=0.3)

    # Create and add a shared legend for detection labels
    detection_patches = [mpatches.Patch(color=label_colors[k], label=label_names[k]) for k in sorted(label_colors)]
    fig.legend(handles=detection_patches, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))

    plt.xlabel('Time Index')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # input shape : 6 x timestamp
    input = np.random.rand(6,80)
    timestamp = np.linspace(0,input.shape[1]/Hz,input.shape[1])

    plot_walk(input, timestamp)
    plot_pedometer(input, timestamp)
    run_plot_pedometer2(input, timestamp)

