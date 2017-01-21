import numpy as np
import time
import subprocess

#######
### K position -- 10
### DIM position -- 11
#######
detector = 'knnofficad'
is_nab = True

detector_file = 'nab_module/nab/article_detectors/' + detector + '/' + detector + '_detector.py'
pre_string_yahoo = 'time python2 nab_module/run.py --skipConfirmation --data Y -n 3 '
pre_string_nab = 'time python2 nab_module/run.py --skipConfirmation --data N -n 6 '

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

k, dim = 3, 1
set_params(k, dim)
make_note(k, dim)

k, dim = 3, 2
set_params(k, dim)
make_note(k, dim)

k, dim = 4, 1
set_params(k, dim)
make_note(k, dim)

k, dim = 4, 2
set_params(k, dim)
make_note(k, dim)

k, dim = 5, 1
set_params(k, dim)
make_note(k, dim)
