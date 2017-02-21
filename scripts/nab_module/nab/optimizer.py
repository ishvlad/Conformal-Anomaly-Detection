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

import numpy as np
import sys
from nab.scorer import scoreCorpus



def optimizeThreshold(args):
  """Optimize the threshold for a given combination of detector and profile.

  @param args       (tuple)   Arguments necessary for the objective function.

  @param tolerance  (float)   Number used to determine when optimization has
                              converged to a sufficiently good score.

  @return (dict) Contains:
        "threshold" (float)   Threshold that returns the largest score from the
                              Objective function.

        "score"     (float)   The score from the objective function given the
                              threshold.
  """
  optimizedThreshold, optimizedScore = twiddle(
    objFunction=objectiveFunction,
    args=args,
    initialGuess=0.1,
    tolerance=1e-7)

  print "Optimizer found a max score of {} with anomaly threshold {}.".format(
    optimizedScore, optimizedThreshold)

  return {
    "threshold": optimizedThreshold,
    "score": optimizedScore
  }


def twiddle(objFunction, args, initialGuess=0.5, tolerance=0.001,
            domain=(0.99, 1.)):
  """Optimize a single parameter given an objective function.

  This is a local hill-climbing algorithm. Here is a simple description of it:
  https://www.youtube.com/watch?v=2uQ2BSzDvXs

  @param args       (tuple)   Arguments necessary for the objective function.

  @param tolerance  (float)   Number used to determine when optimization has
                              converged to a sufficiently good score. Should be
                              very low to yield precise likelihood values.

  @param objFunction(function)Objective Function used to quantify how good a
                              particular parameter choice is.

  @param init       (float)   Initial value of the parameter.

  @param domain     (tuple)   Domain of parameter values, as (min, max).

  @return           (tuple)   Two-tuple, with first item the parameter value
                              yielding the best score, and second item the
                              optimum score from the objective function.
  """
  x = -1
  div_param = 5
  delta = 1. / div_param

  left, right = domain
  while delta > tolerance:
    if x == -1:
      pastCalls = {}
    else:
      try:
        pastCalls = {left: pastCalls[left], right: pastCalls[right]}
      except KeyError:
        pastCalls = {}

    print 'Delta = %f, domain = (%f, %f)' % (delta, left, right)
    sys.stdout.flush()

    for x in np.linspace(left, right, div_param+1):
      if x not in pastCalls:
        pastCalls[x] = np.round(objFunction(x, args), 5)
      print "\tParameter: %f\tScore: %f" % (x, pastCalls[x])
      sys.stdout.flush()

    bestX = max(sorted(pastCalls), key=pastCalls.get)
    bestScore = pastCalls[bestX]
    print 'Delta = %f, Best score: %f' % (delta, bestScore)
    sys.stdout.flush()

    left = max(left, bestX-delta)
    right = min(right, bestX+delta)
    delta = (right - left) / div_param


  # Return the threshold from pastCalls dict. Due to numerical precision, the
  # the while loop may not always yield the threshold that reflects the max
  # score (bestScore).
  return (bestX, bestScore)


def objectiveFunction(threshold, args):
  """Objective function that scores the corpus given a specific threshold.

  @param threshold  (float)   Threshold value to convert an anomaly score value
                              to a detection.

  @param args       (tuple)   Arguments necessary to call scoreHelper.

  @return score     (float)   Score of corpus.
  """
  if not 0 <= threshold <= 1:
    return float("-inf")

  resultsDF = scoreCorpus(threshold, args)
  score = float(resultsDF["Score"].iloc[-1])

  return score
