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

def easyFormat(X, testStart, testLen):
  #Training partitioning
  train = X[:testStart] + X[(testStart + testLen):]
  trainingData = [a[:-1] for a in train]
  trainingTargets = [a[-1] for a in train]
  #Testing partitioning
  test = X[testStart:(testStart + testLen)]
  testData = [a[:-1] for a in test]
  testTargets = [a[-1] for a in test]
    #testData - testing two dimensional set features
    #testTargets - testing onedimensional answer set
    #trainingData - training two dimensional set features
    #trainingTargets - training onedimensional answer set
  return {'trainingData':trainingData, 'trainingTargets':trainingTargets, 'testData':testData, 'testTargets':testTargets}

def fold(X, numFolds, fun):
  grpLen = len(X) / numFolds
  lst = []
  for a in range(numFolds):
    startIndex = a*grpLen
    if a == numFolds - 1:
      grpLen = len(X) - startIndex
    formX = easyFormat(X, a*grpLen, grpLen)
    lst.append(fun(formX))
  average = 0.0
  for a in lst:
    average += a/len(lst)
  return average

