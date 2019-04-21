import numpy

def test(predictions, testSet):
  match = 0.0
  miss = 0.0
  predN = 0
  stop = len(predictions)
  while predN < stop:
    if predictions[predN] == testSet[predN]:
      match += 1
    else:
      miss += 1
    predN = predN + 1
  return match / (match + miss)

def contTest(predictions, testSet):
  predN = 0
  stop = len(predictions)
  average = numpy.average(testSet)
  distanceFromAverage = []
  while predN < stop:
    distanceFromAverage.append(testSet[predN] - average)
    predN += 1
  distanceFromActual = []
  predN = 0
  while predN < stop:
    distanceFromActual.append(testSet[predN] - predictions[predN])
    predN += 1
  sqrdDist = [x*x for x in distanceFromAverage]
  stdDeviation = numpy.average(sqrdDist)
  sqrdDist = [x*x for x in distanceFromActual]
  stdDeviationMine = numpy.average(sqrdDist)
  print('stdDeviation: ' + str(stdDeviation))
  print('devFrmPredic: ' + str(stdDeviationMine))
  return 1

def easyFormat(X, testStart, testLen):
  #Training partitioning
  trainData = numpy.zeros(shape=(len(X) - testLen, len(X[0]) - 1), dtype = float)
  trainData[:testStart] = X[:testStart, :-1]
  trainData[testStart:] = X[(testStart + testLen):, :-1]

  trainTargets = numpy.zeros(shape=(len(X) - testLen), dtype=float)
  trainTargets[:testStart] = X[:testStart, -1]
  trainTargets[testStart:] = X[(testStart + testLen):, -1]

  testData = numpy.zeros(shape=(testLen, len(X[0])-1), dtype = float)
  testData[:] = X[testStart:(testStart + testLen), :-1]

  testTargets = numpy.zeros(shape=(testLen), dtype = float)
  testTargets[:] = X[testStart:(testStart + testLen), -1]

  #testData - testing two dimensional set features
  #testTargets - testing one dimensional answer set
  #trainData - training two dimensional set features
  #trainTargets - training one dimensional answer set
  return {'trainData':trainData, 'trainTargets':trainTargets, 'testData':testData, 'testTargets':testTargets}

def fold(X, numFolds, fun):
  grpLen = int(len(X) / numFolds)
  lst = []
  for a in range(numFolds):
    startIndex = a*grpLen
    if a == numFolds - 1:
      grpLen = len(X) - startIndex
    formX = easyFormat(X, a*grpLen, grpLen)
    lst.append(fun(formX)['accuracy'])
  average = 0.0
  for a in lst:
    average += a/len(lst)
  return average

