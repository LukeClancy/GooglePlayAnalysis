import numpy
import scipy.sparse as spar

THROTTLE=2000

def test(predictions, testSet):
  #for classification not coninuous regression
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

def why(out):
  file = open("out.txt", "a")
  file.write(out)

def contTest(predictions, testSet):
  #compares average distance of test points to guess
  #to average distance of testpoints to average. (std dev)
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
  why(str(stdDeviationMine) + "\n" + str(stdDeviation) + "\n\n")
  return ('stdDeviation: ' + str(stdDeviation), 'devFrmPredic: ' + str(stdDeviationMine))

def sparseWork(X, Y, step, Xstart, Xstop, Ystart, Ystop):
  '''
  Because apparently I dont have enough ram or my graphics card is crap or something.
  Yes I know its stupid. Yes I actually ran into this stupidity.
  :param X: a lil matrix to set to
  :param Y: a lil matrix to grab from
  :param step:
  :param Xstart:
  :param Xstop:
  :param Ystart:
  :param Ystop:
  :return:
  '''
  try:
    assert Ystop - Ystart == Xstop - Xstart
    assert Xstop <= X.shape[0]
    assert Ystop <= Y.shape[0]
  except:
    print(str(step) + " " +str(Xstart) + " "+str(Xstop) + " "+str(Ystart) + " "+str(Ystop) + " ")
    print(str(X.shape))
    print(str(Y.shape))
    print('weird input to sparseWork')
    pass
  length = Ystop - Ystart
  if length <= 0:
    print('len <= 0')
    return
  place = 0
  while place < length:
    if place + step >= length:
      step = length - place
    X[Xstart + place : Xstart + place + step] = Y[Ystart + place: Ystart + place + step]
    place += step

def easyFormat(X, testStart, testLen):
  why("firstLine: " + str(X[0]))
  if spar.issparse(X):
    print('-b')
    dataPoints, dataDim = X.shape
    #seperate the data and targets
    Xtran = X.transpose()
    XData = Xtran[:-1]
    XVal = Xtran[-1]
    print('-ba')
    XData = XData.transpose()
    why('\nAfter Sectioning: ' + str(XData[0]))
    XData.tolil()
    XVal = XVal.transpose().toarray()
    print('-a')
    #seperate data into trainData and testData.
    shp = (dataPoints - testLen, dataDim - 1)
    trainData = spar.lil_matrix(shp, dtype=float)
    print('a')
    sparseWork(trainData, XData, THROTTLE, 0, testStart, 0, testStart)
    #trainData[:testStart] = XData[:testStart]
    print('b')
    sparseWork(trainData, XData, THROTTLE, testStart, shp[0], testStart + testLen, dataPoints)
    #trainData[testStart:] = XData[(testStart + testLen):]
    print('c')
    testData = spar.lil_matrix((testLen, dataDim - 1), dtype=float)
    sparseWork(testData, XData, THROTTLE, 0, testLen, testStart, testStart + testLen)
    #testData[:] = XData[testStart:(testStart + testLen)]

    #Seperate targets
    trainTargets = numpy.append(XVal[:testStart], XVal[testStart + testLen:])
    trainTargets = trainTargets.flatten()
    testTargets = XVal[testStart:(testStart + testLen)].flatten()
    why('\nTTargets: ' + str(testTargets))
  else:
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
  #fold is both broken and unnecessary
  print('in fold')
  length = 0
  if spar.issparse(X):
    length = X.shape[0]
  else:
    length = len(X)
  grpLen = int(length / numFolds)
  lst = []
  for a in range(numFolds):
    startIndex = a*grpLen
    if a == numFolds - 1:
      grpLen = length - startIndex
    formX = easyFormat(X, a*grpLen, grpLen)
    lst.append(fun(formX)['accuracy'])
  average = 0.0
  for a in lst:
    average += a/len(lst)
  return average

