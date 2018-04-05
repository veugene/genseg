from collections import OrderedDict

def parse_log_file(filename):
    dd = OrderedDict({})
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            marker_idx = line.find(":")
            if marker_idx != -1:
                line = line[marker_idx+2::] # Strip "Epoch <num>: "
            contents = line.split(" ")
            contents = [elem.split('=') for elem in contents]
            for tp in contents:
                if tp[0] not in dd:
                    dd[tp[0]] = []
                dd[tp[0]].append(float(tp[1]))
    return dd
