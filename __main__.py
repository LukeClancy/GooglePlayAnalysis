import sklearn
import sklearn.linear_model as lm
import ClancMLBasics as CMB
import sklearn.ensemble as ens
import sklearn.feature_extraction.text as txt
import scipy.sparse
import GPGroups
import GPPredict
import numpy
import sklearn.decomposition as dec
import random
from sklearn.neural_network import MLPRegressor

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
    inputList = numpy.ndarray(shape=shap, dtype=object) #list(data.len(), dtype=object)
    for datPoint in range(len(data)):
        for pos, num in enumerate(inputColumns):
            inputList[datPoint][pos] = data[datPoint][num]
    # calculate targets
    targetList = numpy.ndarray(shape=(len(data),), dtype=float)
    for datPoint in range(len(data)):
        targetList[datPoint] = targetFunction(data[datPoint], lbAtt)
    return (inputList, targetList)

def usableFormat(fileOb):
    dat = []
    line = fileOb.readline()
    columns = line.split('\t')
    for a in range(len(columns)):
        columns[a] = columns[a].strip()
    line =  fileOb.readline()
    while line!=None and line !='':
        nxt = line.split('\t')
        for a in range(len(nxt)):
            nxt[a] = nxt[a].strip()
        dat.append(nxt)
        line = fileOb.readline()
    return (columns, dat)

def changeType(data, column, type):
    if isinstance(column, list):
        column.sort()
        for a in range(len(data)):
            for colN in column:
                if data[a][colN] == 'None' or data[a][colN] == '':
                    data[a][colN] = 0.0
                else:
                    data[a][colN] = type(data[a][colN].replace(",","")) #1,000 -> 1000
        return
    for a in range(len(data)):
        data[a][column] = type(data[a][column])

def catagoriesToNumbers(lib, catagories):
    for cat in catagories:
        if lib.get(cat) == None and cat != '':
            lib[cat] = len(lib.keys())
    return lib

def mlmodel(X, a, maxIter):
    #logr = lm.LinearRegression()
    #logr.fit(X=X['trainData'], y=X['trainTargets'])
    #bayR = lm.BayesianRidge()
    #bayR.fit(X['trainData'], X['trainTargets'])
    #adaBoost = ens.AdaBoostRegressor()
    #adaBoost.fit(X['trainData'], X['trainTargets'])
    NN = MLPRegressor(verbose=True, learning_rate='adaptive', hidden_layer_sizes=(30, 3, 10), max_iter = maxIter,\
		 alpha = a, early_stopping = True, n_iter_no_change=15, \
		activation = 'tanh')
    try:
        print('n-layers:' + str(NN.n_layers))
    except:
        pass
    NN.fit(X['trainData'], X['trainTargets'])
    tstpred = NN.predict(X['testData'])
    return {'out':CMB.contTest(tstpred, X['testTargets']), 'model':NN}

def catagoryExpand(OM_input):
    # format input categories into seperate arrays to break dependacies on catagories
    # change categories to numbers
    catToNum = {}
    for a in range(len(OM_Input[0])):
        catToNum = catagoriesToNumbers(catToNum, OM_Input[:, a])
    # seperate into arrays
    shp = (len(OM_Input), len(catToNum.keys()) * len(OM_Input[0]))  # *2 so we can diffrentiate primary / secondary
    formatted = numpy.zeros(shape=shp, dtype=float)  # +1 for target
    for num in range(len(OM_Input)):
        for catType in range(len(OM_Input[0])):
            if OM_Input[num][catType] != '':
                #set the bit representing both the category, and the category type to one;
                #for the appropriate numbered column.
                formatted[num][catToNum[OM_Input[num][catType]] + len(catToNum) * catType] = 1.0
    return formatted

if __name__ == '__main__':
    myFile = open('dat/Google_Apps_48K.tsv')
    globalAttributes, globalDat = usableFormat(myFile)
    random.shuffle(globalDat)
    #REMOVE ME - too much data for my local computer, run into MemoryError
    globalDat = globalDat[:11000]
    attrLib = asLib(globalAttributes)
    CMB.why('FIRST PRICE: ' + str(globalDat[0][attrLib['app_score']]))
    #------------------------------------------------------------------------------------------------------------------
    #                                       HEY PEOPLE - Data Prediction Selecting Stage
    #-------------------------------------------------------------------------------------------------------------------
    #Pre Processing stage. If you are looking to utilize this research, I would suggest starting here. You can predict many different measures
    #by changing this lamda Function.

    #format types in lambda function that we will be using later (Make them floats).
    changeType(globalDat, [attrLib['app_installs'], attrLib['app_score']], float)
    moneyToFloat = lambda mon: float(mon.replace("$", "").replace("-","0"))
    changeType(globalDat, [attrLib['app_price']], moneyToFloat)

    # This is my first function to calculate opportunity measure.
    # "The average in-app purchase per user is $1.08 for iOS users and $0.43 for Android users, according to an AppsFlyer study." - https://www.braze.com/blog/in-app-purchase-stats/
    # I made the 6 - appscore so it wouldn't be a complete disqualifier
    OM_Func = lambda line, lib: line[lib['app_installs']] * (6.0 - line[lib['app_score']]) * (line[lib['app_price']] + 0.43)
    Like_Func = lambda line, lib: line[lib['app_score']]
    # -------------------------------------------------------------------------------------------------------------------
    #                                       END OF Data Predicting Selection Stage
    # -------------------------------------------------------------------------------------------------------------------
    #Get rid of un-wanted columns and calculate the scores
    OM_Input, OM_Targets = formatForMeasure(globalDat, globalAttributes, ['app_category_primary', 'app_category_secondary'],Like_Func)
    #expand catagories to be mutually independant.
    formatted = catagoryExpand(OM_Input)
    #Get text features
    text = [line[attrLib['app_description']] for line in globalDat]
    toVec = txt.CountVectorizer()
    outVec = toVec.fit_transform(text)
    #cut down on insane amount of text features
    #decomp = dec.PCA(n_components=40000)
    decomp = dec.TruncatedSVD(n_components=int(outVec.shape[1] / 1.01))
    print('ee')
    outVec = decomp.fit(outVec)
    print('a')
    outVec = scipy.sparse.csr_matrix(outVec)
    print('yes')
    #trunc = dec.TruncatedSVD(n_components=outVec.shape[1]/10)
    #trunc.fit(outVec)
    #outVec = trunc.transform(outVec)
    print('description vector shape: ' + str(outVec.shape) + " " + str(type(outVec)))
    #collect the target

    shp = (len(formatted), len(formatted[0]) + 1 + outVec.shape[1])
    withTarget = scipy.sparse.lil_matrix(shp, dtype=float)

    print("WithTarget: " + str(withTarget.shape))
    withTarget = withTarget.transpose()
    print("WithTarget: " + str(withTarget.shape))

    tmp = scipy.sparse.lil_matrix(OM_Targets.reshape((len(OM_Targets), 1))).transpose()
    print("OM_Targets: " + str(OM_Targets.shape))
    withTarget[-1] = tmp

    tmp = scipy.sparse.lil_matrix(formatted).transpose()
    print("formatted: " + str(tmp.shape))
    formattedLen = tmp.shape[0]
    withTarget[:formattedLen] = tmp

    tmp = scipy.sparse.lil_matrix(outVec.transpose())
    print("outVec: " + str(tmp.shape) + ", type: " + str(type(tmp)))
    stop = withTarget.shape[0] - 1
    tmpStop = tmp.shape[0]
    #withTarget[formattedLen:-1] = tmp too big of an opperation, memory error
    try:
        CMB.sparseWork(withTarget, tmp, CMB.THROTTLE, formattedLen, stop, 0, tmpStop)
        #while formattedLen + num < stop:
        #    if num % 10000 == 0: print(num)
        #    withTarget[num + formattedLen] = tmp[num]
        #   num+=1
        #
    except:
        print('sparseWork failure')
        pass


    withTarget = scipy.sparse.csr_matrix(withTarget.transpose())
    print("WithTarget: " + str(withTarget.shape))
    #withTarget[:, withTarget.shape[1] - 1] = OM_Targets
    #ithTarget[:, len(formatted[0])-1] = outVec
    #ithTarget[:,:-(1 + len(outVec[0]))] = formatted

    #test the accuracy
    a = .0001
    maxIter = 1
    final = ''
    X = CMB.easyFormat(withTarget,0, 1001)
    while maxIter < 40:
        maxIter = 150
        while a < .1:
            out = mlmodel(X, a, maxIter)['out']
            final += 'a: '
            final += str(a)
            final += 'maxIt:'
            final += str(maxIter)
            final += ' - '
            final += str(out)
            final += '\n'
            a=a*2
        a = .0001
        maxIter = maxIter * 2
    print('final: ' + final)
