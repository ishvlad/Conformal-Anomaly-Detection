#!/usr/bin/env python
# ----------------------------------------------------------------------
# Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import argparse
import os

try:
    import simplejson as json
except ImportError:
    import json

from nab.runner import Runner
from nab.util import (detectorNameToClass, checkInputs)

# custom detectors
from nab.detectors.null.null_detector import NullDetector

#####################
### article detectors
#####################

from nab.article_detectors.knn.knn_detector import KnnDetector
from nab.article_detectors.knncad.knncad_detector import KnncadDetector
from nab.article_detectors.knnicad.knnicad_detector import KnnicadDetector




# from nab.article_detectors.lof.lof_detector import LofDetector
# from nab.article_detectors.loop.loop_detector import LoopDetector

# from nab.article_detectors.lofcad.lofcad_detector import LofcadDetector
# from nab.article_detectors.loopcad.loopcad_detector import LoopcadDetector


def getDetectorClassConstructors(detectors):
    """
  Takes in names of detectors. Collects class names that correspond to those
  detectors and returns them in a dict. The dict maps detector name to class
  names. Assumes the detectors have been imported.
  """
    detectorConstructors = {
        d: globals()[detectorNameToClass(d)] for d in detectors}

    return detectorConstructors


def main(args):
    numCPUs = int(args.numCPUs) if args.numCPUs is not None else None

    dataDir = args.dataDir
    windowsFile = args.windowsFile
    resultsDir = args.resultsDir
    profilesFile = args.profilesFile
    thresholdsFile = args.thresholdsFile

    runner = Runner(dataDir=dataDir,
                    labelPath=windowsFile,
                    resultsDir=resultsDir,
                    profilesPath=profilesFile,
                    thresholdPath=thresholdsFile,
                    numCPUs=numCPUs)

    runner.initialize()

    if args.detect:
        detectorConstructors = getDetectorClassConstructors(args.detectors)
        runner.detect(detectorConstructors)

    if args.optimize:
        runner.optimize(args.detectors)

    if args.score:
        with open(args.thresholdsFile) as thresholdConfigFile:
            detectorThresholds = json.load(thresholdConfigFile)
        runner.score(args.detectors, detectorThresholds)

    if args.normalize:
        try:
            runner.normalize()
        except AttributeError("Error: you must run the scoring step with the "
                              "normalization step."):
            return


def initialize_args_and_run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--detect",
                        help="Generate detector results but do not analyze results "
                             "files.",
                        default=False,
                        action="store_true")

    parser.add_argument("--optimize",
                        help="Optimize the thresholds for each detector and user "
                             "profile combination",
                        default=False,
                        action="store_true")

    parser.add_argument("--score",
                        help="Analyze results in the results directory",
                        default=False,
                        action="store_true")

    parser.add_argument("--normalize",
                        help="Normalize the final scores",
                        default=False,
                        action="store_true")

    parser.add_argument("--skipConfirmation",
                        help="If specified will skip the user confirmation step",
                        default=False,
                        action="store_true")

    parser.add_argument("--data",
                        help="Y if Yahoo and N if NAB (default)",
                        default="N")

    parser.add_argument("-d", "--detectors",
                        nargs="*",
                        type=str,
                        default=["null", "numenta", "random", "skyline",
                                 "bayesChangePt", "windowedGaussian", "expose",
                                 "relativeEntropy"],
                        help="Comma separated list of detector(s) to use, e.g. "
                             "null,numenta")

    parser.add_argument("-n", "--numCPUs",
                        default=None,
                        help="The number of CPUs to use to run the "
                             "benchmark. If not specified all CPUs will be used.")

    args = parser.parse_args()

    if (not args.detect
        and not args.optimize
        and not args.score
        and not args.normalize):
        args.detect = True
        args.optimize = True
        args.score = True
        args.normalize = True

    if len(args.detectors) == 1:
        # Handle comma-seperated list argument.
        args.detectors = args.detectors[0].split(",")

        # The following imports are necessary for getDetectorClassConstructors to
        # automatically figure out the detector classes.
        # Only import detectors if used so as to avoid unnecessary dependency.
        # if "bayesChangePt" in args.detectors:
        #   from nab.detectors.bayes_changept.bayes_changept_detector import (
        #     BayesChangePtDetector)
        # if "numenta" in args.detectors:
        #   from nab.detectors.numenta.numenta_detector import NumentaDetector
        # if "numentaTM" in args.detectors:
        #   from nab.detectors.numenta.numentaTM_detector import NumentaTMDetector
        # if "null" in args.detectors:
        #   from nab.detectors.null.null_detector import NullDetector
        # if "random" in args.detectors:
        #   import nab.detectors.random.random_detector
        #   global RandomDetector
        # if "skyline" in args.detectors:
        #   from nab.detectors.skyline.skyline_detector import SkylineDetector
        # if "windowedGaussian" in args.detectors:
        #   from nab.detectors.gaussian.windowedGaussian_detector import (
        #     WindowedGaussianDetector)
        # if "relativeEntropy" in args.detectors:
        #   from nab.detectors.relative_entropy.relative_entropy_detector import (
        #     RelativeEntropyDetector)

        # To run expose detector, you must have sklearn version 0.16.1 installed.
        # Higher versions of sklearn may not be compatible with numpy version 1.9.2
        # required to run nupic.
        # if "expose" in args.detectors:
        # from nab.detectors.expose.expose_detector import ExposeDetector

        # if "contextOSE" in args.detectors:
        # from nab.detectors.context_ose.context_ose_detector import (
        # ContextOSEDetector )

    ### Dataset selection
    root = '/'.join(os.path.realpath(__file__).split('/')[:-3])
    if args.data == 'Y':
        args.dataDir = os.path.join(root, 'data/data_yahoo')
        args.windowsFile = os.path.join(root, 'data/labels/yahoo_windows.json')
        args.resultsDir = os.path.join(root, 'experiments/result_yahoo')
        args.thresholdsFile = os.path.join(root, 'experiments/config/thresholds_yahoo.json')
    else:
        args.dataDir = os.path.join(root, 'data/data_nab')
        args.windowsFile = os.path.join(root, 'data/labels/combined_windows.json')
        args.resultsDir = os.path.join(root, 'experiments/result_nab')
        args.thresholdsFile = os.path.join(root, 'experiments/config/thresholds.json')

    args.profilesFile = os.path.join(root, 'experiments/config/profiles.json')

    if args.skipConfirmation or checkInputs(args):
        with open("timing.csv", "w") as myfile:
            myfile.write(args.detectors[0] + ', ' + args.dataDir + '\n')
        main(args)


initialize_args_and_run()
