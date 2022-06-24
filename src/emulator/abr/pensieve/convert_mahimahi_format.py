import argparse
import os

import numpy as np

FILE_SIZE = 2000
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser(
        "Convert bandwidth traces used in simulation to mahimahi format.")
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of the dataset.')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Output directory of the converted traces.')
    return parser.parse_args()


def convert_to_mahimahi_trace(trace_path, out_trace_path):
    with open(trace_path, 'r') as f:
        time_ms = []
        throughput_all = []

        for line in f:
            parse = line.split()
            # trace error, time not monotonically increasing
            if len(time_ms) > 0 and float(parse[0]) < time_ms[-1]:
                raise RuntimeError(
                    'Time decreasing ERROR in TRACE: {}'.format(trace_path))

            time_ms.append(float(parse[0]))
            throughput_all.append(float(parse[1]))

        time_ms = np.array(time_ms)
        throughput_all = np.array(throughput_all)

        with open(out_trace_path, 'w', 1) as mf:
            millisec_time = 0
            mf.write(str(millisec_time) + '\n')
            millisec_count = 0
            last_time = 0
            for i in range(len(throughput_all)):
                throughput = throughput_all[i]

                pkt_per_millisec = throughput*1000/8 / BYTES_PER_PKT

                millisec_count = 0
                pkt_count = 0

                while True:
                    millisec_count += 1
                    millisec_time += 1
                    to_send = (millisec_count * pkt_per_millisec) - pkt_count
                    to_send = np.floor(to_send)

                    for _ in range(int(to_send)):
                        mf.write(str(millisec_time) + '\n')

                    pkt_count += to_send

                    if millisec_count >= (time_ms[i]-last_time)*1000:
                        #millisec_time += 1
                        #mf.write(str(millisec_time) + '\n')
                        last_time = time_ms[i]
                        break
                #mf.write(str(millisec_time) + '\n')


def main():
    args = parse_args()
    data_path = args.root
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(data_path)
    for i, trace_file in enumerate(files):
        if trace_file.startswith('.'):
            continue
        file_path = os.path.join(data_path,  trace_file)
        output_path = os.path.join(output_dir, trace_file)
        convert_to_mahimahi_trace(file_path, output_path)
        print("Converted {}/{} traces...".format(i + 1, len(files)))


if __name__ == '__main__':
    main()
