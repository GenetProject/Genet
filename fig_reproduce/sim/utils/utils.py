import math
import os, glob
import numpy as np

NAMES = ['timestamp', 'bandwidth']


def load_traces_for_train(cooked_trace_folder):
    # print("Loading traces from " + cooked_trace_folder)
    # cooked_files = os.listdir(cooked_trace_folder)
    # print("Found " + str(len(cooked_files)) + " trace files.")
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []

    newest_folder = max( glob.glob( os.path.join( cooked_trace_folder ,'*/' ) ) ,key=os.path.getmtime )
    for subdir ,dirs ,files in os.walk( cooked_trace_folder ):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        if subdir + '/' != newest_folder:
            # sample 0.6*original files out
            random_files = np.random.choice( files ,int( len( files ) * .6 ) )
        else:
            random_files = files

        for file in random_files:
            # print os.path.join(subdir, file)
            file_path = subdir + os.sep + file
            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            #print( val_folder_name, "-----")
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                #print(file_path)
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(val_folder_name + '_' + file)

    return all_cooked_time, all_cooked_bw, all_file_names


# for test
def load_traces(cooked_trace_folder):
    # print("Loading traces from " + cooked_trace_folder)
    # cooked_files = os.listdir(cooked_trace_folder)
    # print("Found " + str(len(cooked_files)) + " trace files.")
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for subdir ,dirs ,files in os.walk( cooked_trace_folder ):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            # print os.path.join(subdir, file)
            file_path = subdir + os.sep + file
            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            #print( val_folder_name, "-----")
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                #print(file_path)
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(val_folder_name + '_' + file)

    return all_cooked_time, all_cooked_bw, all_file_names



def adjust_traces(all_ts, all_bw, bw_noise=0, duration_factor=1):
    new_all_bw = []
    new_all_ts = []
    for trace_ts, trace_bw in zip(all_ts, all_bw):
        duration = trace_ts[-1]
        new_duration = duration_factor * duration
        new_trace_ts = []
        new_trace_bw = []
        for i in range(math.ceil(duration_factor)):
            for t, bw in zip(trace_ts, trace_bw):
                if (t + i * duration) <= new_duration:
                    new_trace_ts.append(t + i * duration)
                    new_trace_bw.append(bw+bw_noise)

        new_all_ts.append(new_trace_ts)
        new_all_bw.append(new_trace_bw)
    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)
    return new_all_ts, new_all_bw


def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf
