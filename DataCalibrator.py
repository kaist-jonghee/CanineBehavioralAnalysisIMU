import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from DataReader import get_labels, df_columns, read_recording, read_imu_file, list_data_code, cleanlabel, all_labels, reverse_all_labels, labels_possible_fields, list_clean_labels, list_data_code_with_labels
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from DataWalkCalibrator import plot_pedometer
from DataWalkCalibrator2 import run_plot_pedometer2, plot_output
from DataPoopCalibrator import DefecationV2Module
import sys
sys.path.append("./ms-tcn")

from scipy.interpolate import interp1d

RecordingsFolder = "Recordings/"
cache_dir = "/home/jonghee/Desktop/UR2025/ms-tcn/cache/"
Hz = 36

def rotate_gravity_seqeuntial(gravity, gyr, ts):
    '''
    (Sequential Rotation Matrices) apply gyroscope data to gravity vector and returns the new gravity vector
    :param gyr: gyroscope data (3,) deg/s
    :param gravity: gravity vector (3,) g
    :param ts: time step (s)
    :return: new gravity vector (3,) g
    '''
    # Gravity Vector always points the same direction(down) with respect to a global frame(coordinate).
    # What the gyr data does is it rotates the sensor frame(coordinate).
    DegreeChangeOfSensorFrame = np.radians(gyr) * ts
    DegreeChangeOfVectors = -DegreeChangeOfSensorFrame
    
    Rx = [[1, 0, 0],[0, np.cos(DegreeChangeOfVectors[0]), -np.sin(DegreeChangeOfVectors[0])],
          [0, np.sin(DegreeChangeOfVectors[0]), np.cos(DegreeChangeOfVectors[0])]]
    Ry = [[np.cos(DegreeChangeOfVectors[1]), 0, np.sin(DegreeChangeOfVectors[1])],
          [0, 1, 0],[-np.sin(DegreeChangeOfVectors[1]), 0, np.cos(DegreeChangeOfVectors[1])]]
    Rz = [[np.cos(DegreeChangeOfVectors[2]), -np.sin(DegreeChangeOfVectors[2]), 0],
          [np.sin(DegreeChangeOfVectors[2]), np.cos(DegreeChangeOfVectors[2]), 0],
          [0, 0, 1]]
    
    Rx = np.array(Rx)
    Ry = np.array(Ry)
    Rz = np.array(Rz)
    
    new_gravity = Rx @ Ry @ Rz @ np.array(gravity).reshape(3)
    return new_gravity
    
def rotate_gravity(gravity, gyro_dps, ts):
    '''
    Apply gyroscope data to gravity vector and returns the new gravity vector
    '''
    
    # Step 1: Convert gyro to radians/sec
    gyro_rad = np.radians(gyro_dps.reshape(3))  # ensure shape (3,)
    
    # Step 2: Compute rotation angle and axis
    angle = np.linalg.norm(gyro_rad) * ts  # total angle rotated in radians
    if angle == 0:
        return gravity  # no rotation

    axis = gyro_rad / np.linalg.norm(gyro_rad)

    # Step 3: Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # Step 4: Rotate gravity vector
    new_gravity = R @ gravity.reshape(3)
    return new_gravity

def rotation_to_z(v):
    '''
    Compute rotation matrix to align vector v with the z-axis
    :param v: input vector (3,)
    :return: rotation matrix (3, 3)
    '''
    v = v.reshape(3)
    target = np.array([0, 0, -1])
    
    # Normalize input
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        raise ValueError("Zero vector cannot be aligned")
    
    v_unit = v / v_norm
    target_unit = target  # Already unit vector

    # Compute axis of rotation (cross product)
    axis = np.cross(v_unit, target_unit)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm == 0:
        # Already aligned or exactly opposite
        if np.allclose(v_unit, target_unit):
            return np.eye(3)  # No rotation needed
        else:
            # 180-degree rotation around any orthogonal axis
            # Choose [1,0,0] unless it's parallel to v
            if np.allclose(v_unit, [0, 0, -1]):
                return np.array([
                    [-1,  0,  0],
                    [ 0, -1,  0],
                    [ 0,  0,  1]
                ])
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v_unit, target_unit), -1.0, 1.0))
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def gravity_to_euler(g):
    '''
    Input:
        g : gravity vector (shape (3,)), assumed normalized (norm = 1)
    Output:
        euler angles (roll, pitch, yaw) in radians
    '''
    g = g.reshape(3)

    # Defensive normalization (optional)
    g = g / np.linalg.norm(g)
    
    # Extract components
    gx, gy, gz = g

    # Compute roll (rotation around x-axis)
    roll = np.arctan2(gy, gz)
    
    # Compute pitch (rotation around y-axis)
    pitch = np.arctan2(-gx, np.sqrt(gy**2 + gz**2))
    
    # Yaw cannot be determined from gravity vector alone
    yaw = 0.0

    return np.array([roll, pitch, yaw])
    
class GravityKalmanFilter:
    def __init__(self, initial_gravity=np.array([0, 0, -1])):
        self.gravity = initial_gravity / np.linalg.norm(initial_gravity)
        
        self.P = np.eye(3) * 0.01  # initial state covariance (small uncertainty)
        self.Q = np.eye(3) * 0.001  # process noise (gyro noise)
        self.R_base = np.eye(3) * 0.02  # base measurement noise (accel noise)
        
    def update(self, gyro_dps, accel_data, ts):
        # Step 1: Prediction
        gravity_predicted = rotate_gravity_seqeuntial(self.gravity, gyro_dps, ts)
        P_predicted = self.P + self.Q
        
        # Step 2: Measurement
        acc_magnitude = np.linalg.norm(accel_data)
        acc_normalized = accel_data / acc_magnitude
        
        # Step 3: Dynamic adjustment of R
        deviation = np.abs(acc_magnitude - 1)  # (g)ab
        if deviation < 0.05:  # stationary
            R = self.R_base  # normal trust in accel
        else:
            # scale = min(deviation / 0.05, 10)  # scale R bigger when moving, cap to 10x
            scale = deviation / 0.05
            R = self.R_base * scale
        
        # Step 4: Innovation
        y = acc_normalized - gravity_predicted  # measurement residual
        S = P_predicted + R  # innovation covariance
        K = P_predicted @ np.linalg.inv(S)  # Kalman gain
        
        # Step 5: Update
        self.gravity = gravity_predicted + K @ y
        self.gravity = self.gravity / np.linalg.norm(self.gravity)  # normalize to unit vector
        
        self.P = (np.eye(3) - K) @ P_predicted
        
        return self.gravity, R, deviation

def calibrate_IMU(df, verbose=True):
    
    acc = df[['aX', 'aY', 'aZ']].to_numpy()
    gyr = df[['gX', 'gY', 'gZ']].to_numpy()
    timestamp = df['timestamp'].to_numpy()
    
    gravity = [acc[0]]
    for g, ts in tqdm(zip(gyr[:-1], np.diff(timestamp)), desc="Processing IMU data-1"):
        gravity.append(rotate_gravity_seqeuntial(gravity[-1], g, ts/1000000))
    gravity = np.array(gravity)
    
    # Kalman Filter
    kalman_filter = GravityKalmanFilter(initial_gravity=acc[0])
    gravity_kalman = [acc[0]]
    if verbose:
        R_kalman = [kalman_filter.R_base[0][0]]
        acc_magnitude = []
    for g, a, ts in tqdm(zip(gyr[:-1], acc[1:], np.diff(timestamp)), total=len(gyr) - 1, desc="Processing IMU data-2"):
        pred_gravity, R, acc_mag = kalman_filter.update(g, a, ts/1000000)
        gravity_kalman.append(pred_gravity)
        if verbose:
            R_kalman.append(R[0][0])
            acc_magnitude.append(acc_mag)
    gravity_kalman = np.array(gravity_kalman)
    if verbose:
        R_kalman = np.array(R_kalman)
        acc_magnitude = np.array(acc_magnitude)
    
    if verbose:
        
        plt.figure(figsize=(12, 8))
        plt.suptitle("Gravity Estimation with Kalman Filter")
        plt.subplot(3,1,1)
        plt.title("aX")
        plt.plot(acc[:,0], c='blue', label='aX')
        # plt.plot(gravity[:,0], c='orange', label='gravityX')
        plt.plot(gravity_kalman[:,0], c='green', label='Kalman-gravityX')
        plt.legend(loc='upper right')
        
        plt.subplot(3,1,2)
        plt.title("aY")
        plt.plot(acc[:,1], c='blue', label='aY')
        # plt.plot(gravity[:,1], c='orange', label='gravityY')
        plt.plot(gravity_kalman[:,1], c='green', label='Kalman-gravityY')
        plt.legend(loc='upper right')
        
        plt.subplot(3,1,3)
        plt.title("aZ")
        plt.plot(acc[:,2], c='blue', label='aZ')
        # plt.plot(gravity[:,2], c='orange', label='gravityZ')
        plt.plot(gravity_kalman[:,2], c='green', label='Kalman-gravityZ')
        plt.legend(loc='upper right')
        
        # plt.subplot(4,1,4)
        # plt.title("gyroscope")
        # plt.plot(gyr)
        # plt.legend(['gX', 'gY', 'gZ'])
        
        plt.tight_layout()
        plt.savefig('gravity_rotation_plot.png')
        plt.close()
    
    if verbose:
        plt.figure(figsize=(12, 8))
        plt.suptitle("Rotating Gravity Vector with Gyroscope Data")
        
        plt.subplot(4,1,1)
        plt.title("Acc")
        plt.plot(acc[:,0], c='blue', label='aX')
        plt.plot(gravity_kalman[:,0], c='green', label='Kalman-gravityX')
        plt.plot(acc[:,1], c='blue', label='aY')
        plt.plot(gravity_kalman[:,1], c='green', label='Kalman-gravityY')
        plt.plot(acc[:,2], c='blue', label='aZ')
        plt.plot(gravity_kalman[:,2], c='green', label='Kalman-gravityZ')
        plt.legend(loc='upper right')
        
        plt.subplot(4,1,2)
        plt.title("gyroscope")
        plt.plot(gyr)
        plt.legend(['gX', 'gY', 'gZ'])
        
        plt.subplot(4,1,3)
        plt.title("R")
        plt.plot(R_kalman, c='blue', label='R scale')
        
        plt.subplot(4,1,4)
        plt.title("acc_magnitude")
        plt.plot(acc_magnitude, c='blue', label='acc_magnitude')
        plt.plot([0.05]*len(acc_magnitude), c='orange', label='threshold')
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
    
    rotation_matrices = [rotation_to_z(gravity) for gravity in gravity_kalman]
    
    cal_acc = np.array([rotation @ acc[i] for i, rotation in enumerate(rotation_matrices)])
    cal_gyr = np.array([rotation @ gyr[i] for i, rotation in enumerate(rotation_matrices)])
    df[['aX', 'aY', 'aZ']] = cal_acc
    df[['gX', 'gY', 'gZ']] = cal_gyr
    
    if verbose:
        plt.cla()
        plt.clf()
        plt.figure(figsize=(12, 8))
        plt.suptitle("Calibrated Accelerometer and Gyroscope Data")
        plt.subplot(2,2,1)
        plt.title("Raw Accelerometer")
        plt.plot(acc[:,0], c='blue', label='aX')
        plt.plot(acc[:,1], c='orange', label='aY')
        plt.plot(acc[:,2], c='green', label='aZ')
        plt.legend(loc='upper right')
        plt.subplot(2,2,3)
        plt.title("Raw Gyroscope")
        plt.plot(gyr[:,0], c='blue', label='gX')
        plt.plot(gyr[:,1], c='orange', label='gY')
        plt.plot(gyr[:,2], c='green', label='gZ')
        plt.legend(loc='upper right')
        
        plt.subplot(2,2,2)
        plt.title("Calibrated Accelerometer")
        plt.plot(cal_acc[:,0], c='blue', label='aX')
        plt.plot(cal_acc[:,1], c='orange', label='aY')
        plt.plot(cal_acc[:,2], c='green', label='aZ')
        plt.legend(loc='upper right')
        plt.subplot(2,2,4)
        plt.title("Calibrated Gyroscope")
        plt.plot(cal_gyr[:,0], c='blue', label='gX')
        plt.plot(cal_gyr[:,1], c='orange', label='gY')
        plt.plot(cal_gyr[:,2], c='green', label='gZ')
        plt.legend(loc='upper right')
        plt.tight_layout()
        # plt.show()
        plt.savefig('calibrated_imu_plot.png')
        plt.close()
    
    return df, gravity_kalman

def upsample_data(data, target_hz=36, original_hz=22):
    """
    data: numpy array of shape (channels, time)
    Returns: upsampled data of shape (channels, new_time)
    """
    channels, time_len = data.shape
    original_t = np.linspace(0, time_len / original_hz, time_len)
    target_t = np.linspace(0, time_len / original_hz, int(time_len * target_hz / original_hz))
    
    upsampled = np.zeros((channels, len(target_t)))
    for i in range(channels):
        f = interp1d(original_t, data[i], kind='linear', fill_value="extrapolate")
        upsampled[i] = f(target_t)
    
    return upsampled

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

def group_numbers(numbers):
    groups = []
    current_group = [numbers[0]]

    for num in numbers[1:]:
        if num - current_group[-1] <= Hz:
            current_group.append(num)
        else:
            if len(current_group) >= 3:
                groups.append([current_group[0], current_group[-1]+1])
            current_group = [num]

    # Add the last group if it has at least 3 elements
    if len(current_group) >= 3:
        groups.append([current_group[0], current_group[-1]+1])

    return groups

def plot_labels(labelslist, title="IMU Data Plot"):
    actions_dict = dict([(label,i) for i,label in enumerate(list_clean_labels())])
    reverse_actions_dict = {v: k for k, v in actions_dict.items()}
    
    tableau_colors = list(mcolors.TABLEAU_COLORS.values())
    css4_colors = list(mcolors.CSS4_COLORS.values())
    colors = tableau_colors + css4_colors[:21 - len(tableau_colors)]
    
    l2c = {label:colors[i] for i,label in enumerate(actions_dict.keys())}
    
    handles = [mpatches.Patch(color=v,label=k) for (k,v) in l2c.items()]
    
    names = [f"Stage {n}" for n in range(len(labelslist))]
    names = ['Original', 'Refined'] if len(labelslist) == 2 else names
    all_seqs = labelslist

    plt.figure(figsize=(6, 8))

    for idx, seq in (enumerate(tqdm(all_seqs))):
        bkp = list(np.array(seq[1:]) == np.array(seq[:-1]))
        i = 0
        xname = names[idx]
        seq = list(seq)
        
        while False in bkp:
            feat = seq[0]
            bp = bkp.index(False)
            count = len(seq[:bp+1])
            bkp = bkp[bp+1:]
            seq = seq[bp+1:]
            feat = feat.split('_')[0] if isinstance(feat, str) else feat  # in case real is string-labeled
            plt.bar(xname, count, bottom=i, color=l2c[feat], label=feat, width=0.3)
            i += count
        if len(bkp):
            feat = seq[0]
            count = len(bkp)
            feat = feat.split('_')[0] if isinstance(feat, str) else feat
            plt.bar(xname, count, bottom=i, color=l2c[feat], label=feat, width=0.3)

    # Remove duplicate labels in legend
    handles_dict = {}
    for h in handles:
        handles_dict[h.get_label()] = h
    handles_unique = list(handles_dict.values())

    plt.legend(handles=handles_unique, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    print("saving plot to", os.path.join('/home/jonghee/Desktop/UR2025/images', title + ".png"))
    plt.savefig(os.path.join('/home/jonghee/Desktop/UR2025/images', title + ".png"))
    # plt.show()
    plt.cla()
    return

def find_overlapping_ranges(ranges1, ranges2):
    """
    Finds overlapping regions between ranges1 and ranges2.

    Args:
        ranges1: list of [start, end] pairs (inclusive start, exclusive end)
        ranges2: list of [start, end] pairs (inclusive start, exclusive end)

    Returns:
        List of [ [start, end], overlap_length ] for each overlapping pair
    """
    overlaps = []
    for s1, e1 in ranges1:
        for s2, e2 in ranges2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start < end:  # valid overlap
                overlaps.append([[s1, e1], end - start])
    return overlaps    

def refine_label_with_detections(detections, defidx, Hz, possible_labels=[0]):
    """
    Refine the labels based on stationary segments.
    """
    if len(defidx) == 0 or len(possible_labels) == 0:
        return []
    
    isStat = np.isin(detections, possible_labels)
    isStat = find_continuous_ones(isStat)
    defidx = group_numbers(defidx)
    
    newLabelRanges = []
    
    for [defstart, defend] in defidx:
        candranges = find_overlapping_ranges(isStat, [[defstart, defend]])
        if len(candranges) == 0:
            candStat = isStat.copy()
            candStat = sorted(candStat, key=lambda x: defstart-x[1] if defstart>=x[1] else x[0] - defend)
            if len(candStat) == 0:
                newLabelRanges.append([defstart, defend])
            elif candStat[0][0] - defend <= Hz or candStat[0][1] - defstart <= Hz:
                # If the candidate range is too close to the defecation range, use it
                newLabelRanges.append([candStat[0][0], candStat[0][1]])
            else:
                newLabelRanges.append([defstart, defend])
        else:
            candranges = sorted(candranges, key=lambda x: x[1], reverse=True)
            found = False
            # print(f"Defecation Range: {defstart}-{defend}, Candidate Ranges: {candranges}")
            for [start, end], count in candranges:
                if count >= (defend-defstart)/2: # Overlaps with majority of the OG labels.
                    found = True
                    if (end-start)*0.6 <= count: # If the overlapping region accounts for more than 60% of the new found region, use the entire new found region.
                        newLabelRanges.append([start, end])
                    else:
                        newLabelRanges.append([max(defstart,start), min(end,defend)]) # Only the overlapping part
                elif (end-start)*0.6 <= count: # the range is included (60% above)
                    found = True
                    newLabelRanges.append([start, end])
            if not found:
                candranges = sorted(candranges, key=lambda x: x[1], reverse=True)
                newLabelRanges.append([candranges[0][0][0], candranges[0][0][1]])
    
    return newLabelRanges

def label_refinement(data_code, verbose=False):
    if os.path.exists(os.path.join(cache_dir, f"{data_code}_DATA.npy")) and os.path.exists(os.path.join(cache_dir, f"{data_code}_output.npy")):
        data = np.load(os.path.join(cache_dir, f"{data_code}_DATA.npy"), allow_pickle=True)
        output = np.load(os.path.join(cache_dir, f"{data_code}_output.npy"), allow_pickle=True).item()
        detections = np.array(output['detections'])  # Ensure it's a NumPy array
        imu = data[:6]
    else:
        df = read_imu_file(data_code)
        if len(df) < 36:
            return
        df, gravity = calibrate_IMU(df, verbose=False)
        data = np.concatenate([df[['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']].to_numpy().T,gravity.T])
        if data_code.startswith("PUMPKIN_"):
            data = upsample_data(data, target_hz=Hz, original_hz=25)
            if data_code.startswith('PUMPKIN_T'):
                # data = data.transpose(0,1)
                # data = data.transpose(3,4)
                data = data[[1,0,2,4,3,5,7,6,8]]
                data[0] = -data[0]
                data[1] = -data[1]
                data[3] = -data[3]
                data[4] = -data[4]
                data[6] = -data[6]
                data[7] = -data[7]
        imu = data[:6]
        
        # ** output 설명 **
        # 0: Stationary, 1: walk, 2: run, 3: jump, 4: shake, 5: scratch
        # 'detections': datapoint크기의 리스트, 각 아이템 == detected 행동,
        # 'walk_pedometer': 걷기의 시작-끝 chunk에서의 걸음 수,
        # 'pedometer_time': 각 step을 한 시간 (time) ,
        # 'curve_startend':curve를 했다면, 그 시작과 끝 pair의 list (time),
        # 'curve_degree':curve의 시작-끝 chunk에서의 curve degree}
        output = run_plot_pedometer2(imu, np.array(range(imu.shape[-1])))
        detections = np.array(output['detections'])  # Ensure it's a NumPy array
        
        # Build a cache('.npy') to save data,labels at once
        # np.save(os.path.join(cache_dir, f"{data_code}_DATA.npy"), np.array(data,dtype=object))
        # np.save(os.path.join(cache_dir, f"{data_code}_output.npy"), np.array(output,dtype=object))
    
    if os.path.exists(os.path.join(cache_dir, f"{data_code}_LABELS.npy")) and os.path.exists(os.path.join(cache_dir, f"{data_code}_LABELS_REFINED.npy")):
        og_labels = np.load(os.path.join(cache_dir, f"{data_code}_LABELS.npy"), allow_pickle=True)
        refined_labels = np.load(os.path.join(cache_dir, f"{data_code}_LABELS_REFINED.npy"), allow_pickle=True)
    else:
        df = read_imu_file(data_code)
        labels = df['labels'].to_numpy()
        possible_labels = list(set(labels))
        possible_labels = list(set([all_labels['nan'] if isinstance(v, float) and np.isnan(v) else all_labels[v] for v in possible_labels]))
        
        labels_original = labels.copy()
        labels = np.round(np.linspace(0, len(labels)-1, data.shape[1])).astype(int)
        labels = labels[labels < len(labels_original)]
        labels = labels_original[labels]  # remap labels to new time axis
        
        refined_labels = np.array([all_labels['nan'] if isinstance(l, float) and np.isnan(l) else all_labels[l] for l in labels], dtype=object)
        # refined_labels = np.full(len(labels), "Unlabeled", dtype=object)  # Initialize with "Unlabeled"
        refined_labels[np.where(detections == 2)[0]] = "Run"
        refined_labels[np.where(np.isin(detections,[4,5]))[0]] = "Shake"
        
        og_labels = np.array([all_labels['nan'] if isinstance(l, float) and np.isnan(l) else all_labels[l]  for l in labels])
        
        for pl in possible_labels:
            if len(labels_possible_fields[pl]) == 0:
                continue
            defidx = np.where(np.isin(labels,reverse_all_labels[pl]))[0]
            newLabelRanges = refine_label_with_detections(detections, defidx, Hz, possible_labels=labels_possible_fields[pl]) # Selecting which pre-segmented region to follow.
            
            # print(pl, newLabelRanges, group_numbers(defidx))
            
            # What should we do if the selected region overlaps?
            # First get rid of the previous label
            refined_labels[np.where(refined_labels == pl)[0]] = "Unlabeled"
            
            for [start,end] in newLabelRanges:
                present_labels = list(set(refined_labels[start:end]))
                if len(present_labels) == 1 and present_labels[0] == "Unlabeled": # Nice~
                    refined_labels[start:end] = [pl]*(end-start)
                else:
                    # Check if we can find the boundary from our pl label.
                    plbool = og_labels == pl
                    if sum(plbool[start:end]) > 0: # Boundary is inside the range
                        idx = np.arange(start, end)[plbool[start:end].astype(bool)]
                        refined_labels[idx] = [pl]*len(idx)
                    else: # no boundary found
                        if "Unlabeled" in present_labels:
                            present_labels.remove("Unlabeled")
                        idxbool = np.ones((end-start,))
                        for l in present_labels: # Check if other labels present in the range can provide boundaries
                            lbool = og_labels == l
                            if sum(lbool[start:end]) > 0: # If they do have boundaries, set it
                                refined_labels[start:end][np.where(refined_labels[start:end] == l)[0]] = ["Unlabeled"] * len(np.where(refined_labels[start:end] == l)[0])
                                refined_labels[np.arange(start,end)[lbool[start:end].astype(bool)]] = [l]*sum(lbool[start:end])
                                
                                idxbool = idxbool * ~lbool[start:end] # Remove the boundary from idx
                                
                        if sum(idxbool) == 0:
                            pass
                            # print("Couldn't find a proper region for label", pl, "from", start, "to", end)
                        else:
                            idx = np.arange(start, end)[idxbool.astype(bool)]
                            if len(idx) > 0:
                                roomforchange = np.where(refined_labels[idx] == "Unlabeled")[0]
                                if len(roomforchange) == 0:
                                    # print("CONFICTS FOUND, changing to UNLABELED")
                                    refined_labels[idx] = ["Unlabeled"] * len(idx)
                                else:
                                    refined_labels[idx[roomforchange]] = [pl] * len(roomforchange)
                                    # print("Found", len(roomforchange), "points to change to label", pl)
                            else:
                                pass
                                # print("Forced this label to not fit in the range", start, "to", end)
                                
                
                # refined_labels[start:end] = [pl]*(end-start)
            
            if verbose and len(defidx) > 0:
                defidx = group_numbers(defidx)
                defidx3 = newLabelRanges
                
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

                fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
                
                # Add background color to both subplots
                for ax in axes[:2]:
                    for i in range(len(detections)):
                        ax.axvspan(i - 0.5, i + 0.5, color=color_array[i], alpha=0.3)

                # Plot accelerometer
                axes[0].set_title('Accelerometer (aX, aY, aZ)')
                for i in range(3):
                    axes[0].plot(imu[i, :], label=f'a{i}')
                for [start,end] in defidx:
                    axes[0].scatter(list(range(start,end)), imu[0, start:end], color='red', label='Label')
                axes[0].legend(loc='upper right')

                # Plot gyroscope
                axes[1].set_title('Gyroscope (gX, gY, gZ)')
                for i in range(3, 6):
                    axes[1].plot(imu[i, :], label=f'g{i-3}')
                for [start,end] in defidx3:
                    axes[1].scatter(list(range(start,end)), imu[5, start:end], color='red', label='Label Refined')
                axes[1].legend(loc='upper right')
                
                # Create and add a shared legend for detection labels
                detection_patches = [mpatches.Patch(color=label_colors[k], label=label_names[k]) for k in sorted(label_colors)]
                fig.legend(handles=detection_patches, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.5))

                
                
                # Plot accelerometer
                axes[2].set_title('Accelerometer (aX, aY, aZ)')
                for i in range(3):
                    axes[2].plot(imu[i, :], label=f'a{i}')
                axes[2].legend(loc='upper right')

                # Plot gyroscope
                axes[3].set_title('Gyroscope (gX, gY, gZ)')
                for i in range(3, 6):
                    axes[3].plot(imu[i, :], label=f'g{i-3}')
                axes[3].legend(loc='upper right')

                

                plt.xlabel('Time Index')
                plt.tight_layout()
                # plt.savefig('test.png')
                # plt.show()
                # plt.savefig(f"{data_code}_LABELS_REFINED.png")
                plt.cla()
                plt.close()
        
        og_labels = np.array([all_labels['nan'] if isinstance(l, float) and np.isnan(l) else all_labels[l]  for l in labels])
        
        print("Overlap Percentage:", data_code, np.sum(og_labels == refined_labels) / len(og_labels) * 100, "%")
    
        # np.save(os.path.join(cache_dir, f"{data_code}_LABELS.npy"), np.array(og_labels,dtype=object))
        # np.save(os.path.join(cache_dir, f"{data_code}_LABELS_REFINED.npy"), np.array(refined_labels,dtype=object))
    
    plot_labels([og_labels, refined_labels])
    return og_labels, refined_labels




if __name__ == "__main__":
    # files = os.listdir(RecordingsFolder)
    # for file in files:
    # # file = 'Formatted_0507_cal4.csv'
    #     if file.endswith(".csv"):
    #         data = read_recording(file)
    #         if data is not None:
    #             print(f"Loaded {file} with {len(data)} rows.")
    #         else:
    #             print(f"Failed to load {file}.")
    #         calibrate_IMU(data)
    # assert False
    # data_codes = list_data_code_with_labels()

    # for dc in data_codes:
    #     df = read_imu_file(dc)
    #     ls = get_labels(df)
    #     # if "Shake" in ls or "Scratch" in ls:
    #     # if "Scratch" in ls:
    #     if "Eat" in ls:
    #         print(dc, len(df))
    # print("done")
    # assert False
    
    # data_code = "ARDUINO_S22"
    # data_code = "ARDUINO_N63"
    data_code = "PUMPKIN_A51"
    # data_code = "PUMPKIN_A71"
    # data_code = "PUMPKIN_E60"
    
    # region: POOP
    # ARDUINO_S47 26884
    # PUMPKIN_A33 1060
    # PUMPKIN_A41 920
    # PUMPKIN_A52 940
    # PUMPKIN_A74 1240
    # ARDUINO_S11 44468
    # PUMPKIN_A19 800
    # PUMPKIN_A9 1160
    # PUMPKIN_A65 960
    # PUMPKIN_A62 1212
    # PUMPKIN_A77 1080
    # PUMPKIN_A84 740
    # PUMPKIN_A15 2318
    # PUMPKIN_A43 840
    # PUMPKIN_A21 877
    # PUMPKIN_A71 1110
    # PUMPKIN_A82 593
    # done
    # endregion
    
    # region: URINATE
    # PUMPKIN_A60 1200
    # PUMPKIN_A51 320
    # PUMPKIN_A59 640
    # PUMPKIN_A45 920
    # PUMPKIN_A1 320
    # PUMPKIN_A4 220
    # PUMPKIN_A66 360
    # PUMPKIN_A27 723
    # PUMPKIN_A64 2240
    # PUMPKIN_A83 1171
    # ARDUINO_S14 20604
    # PUMPKIN_A57 460
    # PUMPKIN_A26 360
    # PUMPKIN_A17 510
    # PUMPKIN_A86 540
    # PUMPKIN_A79 912
    # PUMPKIN_A47 420
    # PUMPKIN_A50 920
    # PUMPKIN_A3 180
    # PUMPKIN_A78 780
    # PUMPKIN_A5 340
    # PUMPKIN_A46 440
    # PUMPKIN_A123 31542
    # PUMPKIN_A85 470
    # PUMPKIN_A70 1813
    # PUMPKIN_A24 461
    # PUMPKIN_A29 1056
    # PUMPKIN_A98 45800
    # PUMPKIN_A32 280
    # PUMPKIN_A67 580
    # PUMPKIN_A28 1115
    # PUMPKIN_A18 754
    # PUMPKIN_A81 440
    # PUMPKIN_A75 582
    # PUMPKIN_A69 1180
    # done
    # endregion
    
    for data_code in tqdm(list_data_code_with_labels()[1:]):
        if data_code == "PUMPKIN_E695":
            continue
        if data_code == "PUMPKIN_E1004":
            continue
        if data_code == "PUMPKIN_E827":
            continue
        # print(data_code)
        data_code = "ARDUINO_S47"
        label_refinement(data_code, verbose=True)
        break

    # df = read_imu_file(data_code)    
    # df, gravity = calibrate_IMU(df, verbose=False)
    # data = df[['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ', 'timestamp']].to_numpy().T
    # # data = np.concatenate([gravity.T, df[['gX', 'gY', 'gZ', 'timestamp']].to_numpy().T])
    # newdata = upsample_data(data, target_hz=Hz, original_hz=25)
    # # newdata = data
    # imu = newdata[:6]
    # # Walk Signal Processing
    # plot_pedometer(imu, np.array(range(imu.shape[-1])))
    