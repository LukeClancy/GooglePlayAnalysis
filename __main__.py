import sklearn
import sklearn.linear_model as lm
import ClancMLBasics as CMB
import sklearn.ensemble as ens
import sklearn.feature_extraction.text as txt
import scipy.sparse
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

def mlmodel(X):
    #logr = lm.LinearRegression()
    #logr.fit(X=X['trainData'], y=X['trainTargets'])
    #bayR = lm.BayesianRidge()
    #bayR.fit(X['trainData'], X['trainTargets'])


    adaBoost = ens.AdaBoostRegressor()
    adaBoost.fit(X['trainData'], X['trainTargets'])
    tstpred = adaBoost.predict(X['testData'])
    return {'accuracy':CMB.contTest(tstpred, X['testTargets']), 'model':adaBoost}

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
    attrLib = asLib(globalAttributes)
    #-------------------------------------------------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------------------------------------------------
    #                                       END OF Data Predicting Selection Stage
    # -------------------------------------------------------------------------------------------------------------------
    #Get rid of un-wanted columns and calculate the scores
    OM_Input, OM_Targets = formatForMeasure(globalDat, globalAttributes, ['app_category_primary', 'app_category_secondary'], OM_Func)
    #expand catagories to be mutually independant.
    formatted = catagoryExpand(OM_Input)
    #Add text features
    text = [line[attrLib['app_description']] for line in globalDat]
    toVec = txt.CountVectorizer()
    outVec = toVec.fit_transform(text)
    print('description vector shape: ' + str(outVec.shape) + " " + str(type(outVec)))
    #collect the target
    shp = (len(formatted), len(formatted[0]) + 1 + outVec.shape[1])
    withTarget = scipy.sparse.csc_matrix(shp, dtype=float)
    withTarget[:][withTarget.shape[1] - 1] = OM_Targets

    for a in range(withTarget.shape[0]):
        withTarget[a][0] = OM_Targets[a]
    withTarget[:, len(formatted[0]):-1] = outVec
    withTarget[:,:-(1 + len(outVec[0]))] = formatted

    #test the accuracy
    numpy.random.shuffle(withTarget)
    accuracy = CMB.fold(withTarget, 10, mlmodel)
    print('accuracy of formattedData: ' + accuracy)
    pass
