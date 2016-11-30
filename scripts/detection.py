import numpy as np
import time
import subprocess
import sys

#######
### K position -- 17
### DIM position -- 18
detector = 'loopcad'
detector_file = 'nab/detectors/'+detector+'/'+detector+'_detector.py'
pre_string_yahoo = 'time python2 run.py -n 8 --dataDir data_yahoo --windowsFile labels/yahoo_windows.json --resultsDir result_yahoo --skipConfirmation '
pre_string_nab = 'time  python2 run.py -n 8 --skipConfirmation ' 
threshold_pattern = '{\n\t"'+detector+ """": {
        "reward_low_FN_rate": {
            "score": 0.0,
            "threshold": %f
        },
        "reward_low_FP_rate": {
            "score": 0.0,
            "threshold": %f
        },
        "standard": {
            "score": 0.0,
            "threshold": %f
        }
    }
}"""

################ 
## Parameters Generation
################
def get_k_dim(known=None, k_first=True):
    params = [4,9,14,19,24,30]
    if known is None:
    	known = np.random.randint(4, 30)
    if k_first:
        return np.vstack((np.repeat(known, len(params)), params)).T
    else:
        return np.vstack((params, np.repeat(known, len(params)))).T

def set_params(k, dim):
	text = None
	with open(detector_file, "r") as f:
		text = f.readlines()
	text[16] = '        self.k = '+str(k)+'\n'
	text[17] = '        self.dim = '+str(dim)+'\n'
	with open(detector_file, "w") as f:
		f.writelines(text)

################ 
## Detection
################
def make_note(k, dim, is_nab):
	threshold_file = 'config/thresholds.json' if is_nab else 'config/thresholds_yahoo.json'
	string = pre_string_nab if is_nab else pre_string_yahoo + ' --thresholdsFile '+threshold_file
	string += ' --score --normalize -d ' + detector
	result_file = detector+"_nab_res.txt" if is_nab else detector+"_yahoo_res.txt"
	result_file = 'opt/'+result_file

	#thresholds = np.linspace(0.88,1.,13)
	thresholds = np.array([0.9,0.925,0.95,0.96,0.97,0.98,0.99,0.9925,0.995,0.996,0.997,1.0])
	results = []

	with open(result_file, "a") as f:
	        wr = '-----------------------------\n\t'
	        wr += 'NAB ' if is_nab else 'Yahoo '
	        wr += detector + '(%d,%d)\n' % (k, dim)
	        f.write(wr)
	for t in thresholds:
	    with open(threshold_file, 'w+') as f:
	        f.write(threshold_pattern % (t,t,t))
	    res = subprocess.check_output(string[5:],stderr=subprocess.STDOUT,shell=True)
	    with open(result_file, "a") as f:
	    	arr = str(res).split("\n", 10)[-4:-1]
	    	result = list(map(lambda s: str(float(s.split(' ',15)[-1])),arr))
	        results.append(result)
	        wr = str(t)+', '+', '.join(results[-1])+', '+time.ctime()+'\n'
	        f.write(wr)
	    time.sleep(5)

	with open(result_file, "a") as f:
		idx = np.argmax(np.array(results).astype(float), axis=0)
		best = '\nBEST:\t'+str(k)+'\t'+str(dim)+'\t|\t'+' '.join([results[idx[i]][i] for i in range(3)]) + ' | '+ ' '.join(thresholds[idx].astype(str)) + '\n'
		f.write(best)

count = 4
add = 4
for _ in range(1):
    params = get_k_dim(known=20, k_first=False)
	#params = [[8,3],[8,5],[10,3],[10,5]]
    for k,dim in params:
		set_params(k, dim)

		# Yahoo
		print str(count)+': Start detect '+detector+'('+str(k)+', '+str(dim)+') Yahoo..',
		sys.stdout.flush()
		detect_string = pre_string_yahoo + ' --detect  -d ' +detector
		subprocess.check_output(detect_string,stderr=subprocess.STDOUT,shell=True)
		print 'Done. Now optimize..',
		sys.stdout.flush()
		make_note(k, dim, is_nab=False)
		print 'OK.'
		sys.stdout.flush()
		count+=add
		
		dim -= 6
		# NAB
		print str(count)+': Start detect '+detector+'('+str(k)+', '+str(dim)+') NAB..',
		sys.stdout.flush()
		detect_string = pre_string_nab + ' --detect  -d ' +detector
		subprocess.check_output(detect_string,stderr=subprocess.STDOUT,shell=True)
		print 'Done. Now optimize..',
		sys.stdout.flush()
		make_note(k, dim, is_nab=True)
		print 'OK.'		
		sys.stdout.flush()
		count+=add

