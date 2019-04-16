import sklearn
import sklearn.ensemble as skens
import ClancMLBasics as CMB
import GPGroups
import GPPredict
import numpy

#app_id	app_url	app_name	app_category_primary	app_category_secondary	app_score	app_rating_5
#app_rating_4	app_rating_3	app_rating_2	app_rating_1	app_no_reviews	app_description	app_is_editors_choice
#app_price 	app_is_free	app_in_app_purchase	app_last_updated	app_size	 app_size_MB 	app_installs
#app_current_version	app_required_version	app_content_rating	developer_id	developer_name	developer_url
#developer_address	developer_identifier

#two parts, general groupings and specific prediction

asLib = lambda list: {list[loc]:loc for loc in range(len(list))}
globalAttributes = []
globalDat = []

def _getNumAttributeColumnNums(lib, inputStrs):
    out = []
    for a in inputStrs:
        out.append(lib[a])
    out.sort()
    return out

def formatForMeasure(data, attributes, learningInputs, targetFunction):
    '''
    :param data: is data with all the columns attached
    :param attributes: a list representing the meaning of the columns
    :param learningInputs: the inputs of the learning algorithm
    :param targetFunction: a function with inputs (list, library) -> float where the library is of the form 'attribute -> column number'. \
        'library' is auto-generated within the function from 'attributes'
    :return: (learningInputList, TargetValuesList)
    '''
    #get input list, by getting column num from learningInputs, and referencing lbAtt
    lbAtt = asLib(attributes)
    inputColumns = _getNumAttributeColumnNums(lbAtt, learningInputs)
    shap = (len(data), len(learningInputs))
    inputList = numpy.ndarray(shape=shap, dtype=float) #list(data.len(), dtype=object)

    for datPoint in range(data.len()):
        for num in inputColumns:
            assert type(data[datPoint][num]) is float
            inputList[datPoint][num] = data[datPoint][num]
    # calculate targets
    targetList = numpy.ndarray(shape=(data.len(),), dtype=float)
    for datPoint in range(data.len()):
        targetList[datPoint] = targetFunction(data[datPoint], lbAtt)
    return (inputList, targetList)

def usableFormat(fileOb):
    dat = []
    line = fileOb.readline()
    columns = line.split('\t')
    for a in range(len(columns)):
        columns[a] = columns[a].trim()
    line =  fileOb.readline()
    while line!=None and line !='':
        nxt = line.split('\t')
        for a in range(len(nxt)):
            nxt[a] = nxt[a].strip()
        dat.append(nxt)
    return (columns, dat)

def changeType(data, column, type):
    if type(column) is list:
        column.sort()
        for a in range(len(data)):
            for colN in column:
                data[a][colN] = type(data[a][colN])
        return
    for a in range(len(data)):
        data[a][column] = type(data[a][column])

def catagoriesToNumbers(lib, catagories):
    for cat in catagories:
        if lib.get(cat) == None:
            lib[cat] = len(lib.keys())
    return lib

def ranF(X):
    randomForest = skens.RandomForestClassifier()
    randomForest.fit(X['trainingData'], X['trainingTargets'])
    tstpred = randomForest.predict(X['testData'])
    return CMB.test(tstpred, X['testTargets'])

def testAccuracies(formattedData):
    accuracy = CMB.fold(formatted, 10, ranF)
    print('accuracy of formattedData: ' + accuracy)

if __name__ == '__main__':
    myFile = open('dat/Google_Apps_48K.tsv')
    globalAttributes, globalDat = usableFormat(myFile)
    attrLib = asLib(globalAttributes)

    #format types in lambda function
    changeType(globalDat, [attrLib['app_installs'], attrLib['app_score']], float)
    moneyToFloat = lambda mon: float(mon.replace("$", ""))
    changeType(globalDat, [attrLib['app_price']], moneyToFloat)

    OM_Func = lambda line, lib: line[lib['app_installs']] * (5.0 - line[lib['app_score']]) * line[lib['app_price']]
    OM_Input, OM_Targets = formatForMeasure(globalDat, globalAttributes, ['app_category_primary', 'app_category_secondary'], OM_Func)

    #format input categories into seperate arrays to break dependacies on catagories
        #change categories to numbers
    catToNum = {}
    catToNum = catagoriesToNumbers(catToNum, OM_Input[:,0])
    catToNum = catagoriesToNumbers(catToNum, OM_Input[:,1])
        #seperate into arrays
    shp = numpy.shape((len(OM_Input), len(catToNum) * 2 + 1))   #*2 so we can diffrentiate primary / secondary
    formatted = numpy.zeros(shape=shp, dtype=float)            #+1 for target
    for num in range(len(OM_Input)):
        formatted[num][catToNum[OM_Input[0]]] = 1
        formatted[num][catToNum[OM_Input[1]] + len(catToNum)] = 1
        formatted[num][-1] = OM_Targets[num]

    #test the accuracy
    testAccuracies(formatted)

