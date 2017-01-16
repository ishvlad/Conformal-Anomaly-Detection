import numpy as np
import time
import subprocess

#######
### K position -- 10
### DIM position -- 11
#######
detector = 'knn'
is_nab = True

detector_file = 'nab_module/nab/article_detectors/' + detector + '/' + detector + '_detector.py'
pre_string_yahoo = 'time python nab_module/run.py --skipConfirmation --data Y -n 4'
pre_string_nab = 'time python nab_module/run.py --skipConfirmation --data N -n 4 '

result_file = 'nab_' if is_nab else 'yahoo_'
result_file += detector + '.txt'
result_file = '../experiments/temp/' + result_file


################
## Parameters Generation
################

def set_params(k, dim):
    with open(detector_file, "r") as f:
        text = f.readlines()
    text[9] = '        self.k = ' + str(k) + '\n'
    text[10] = '        self.dim = ' + str(dim) + '\n'
    with open(detector_file, "w") as f:
        f.writelines(text)


################
## Detection
################
def make_note(k, dim):
    string = pre_string_nab if is_nab else pre_string_yahoo
    string += ' -d ' + detector

    # Detect + Score + Threshold optimize
    output_string = subprocess.check_output(string[5:], stderr=subprocess.STDOUT, shell=True)
    scores = str(output_string).split("\n")[-5:-2]
    scores = list(map(lambda s: str(float(s.split(' ', 15)[-1])), scores))
    final_score = sum(list(map(float, scores)))

    write_string = '(%d,%d)' % (k, dim) + \
                   '\t' + ', '.join(scores) + ' | ' + \
                   str(final_score) + \
                   '\t' + time.ctime() + '\n'
    with open(result_file, "a") as f:
        f.write(write_string)

    return final_score


ks =  np.linspace(2, 11, 5).astype(int)
dims = np.linspace(2, 11, 5).astype(int)

# edit division
to_tuple = lambda ij: (ij / len(dims), ij % len(dims))
buffer = [177.35, 137.93, np.inf, np.inf]

while len(ks) > 3 or len(dims) > 3:
    vals = np.ones((len(ks), len(dims))) * np.inf
    if len(buffer) != 0:
        vals[[0, 0, -1, -1], [0, -1, 0, -1]] = buffer

    status_string = 'k: ' + ', '.join(ks.astype(str)) + \
                    '\tdim: ' + ', '.join(dims.astype(str)) + '\t' + time.ctime() + '\n'
    with open(result_file, "a") as f:
        f.write(status_string)

    for i, k in enumerate(ks):
        for j, dim in enumerate(dims):
            if vals[i, j] == np.inf:
                set_params(k, dim)
                vals[i, j] = make_note(k, dim)
                time.sleep(100)

    best_ind = to_tuple(np.argmax(vals))
    best = (ks[best_ind[0]], dims[best_ind[1]])

    min_vals = (max(best_ind[0] - 1, 0), max(best_ind[1] - 1, 0))
    max_vals = (min(best_ind[0] + 1, len(ks) - 1), min(best_ind[1] + 1, len(dims) - 1))

    buffer = [vals[min_vals], vals[min_vals[0], max_vals[1]],
              vals[max_vals[0], min_vals[1]], vals[max_vals]]

    ks = np.array(sorted(list(set(np.linspace(ks[min_vals[0]], ks[max_vals[0]], 5).astype(int)))))
    dims = np.array(sorted(list(set(np.linspace(dims[min_vals[1]], dims[max_vals[1]], 5).astype(int)))))

    status_string = 'BEST:\t' + str(vals[best_ind]) + ' at ' + str(best) + '\n' + \
                    '-'*40 + '\n'
    with open(result_file, "a") as f:
        f.write(status_string)

with open(result_file, "a") as f:
    f.write('STOP ITERATION\t' + time.ctime() + '\n\n')
