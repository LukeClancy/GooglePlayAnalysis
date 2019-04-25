import sklearn
import sklearn.linear_model as lm
import ClancMLBasics as CMB
import sklearn.ensemble as ens
import sklearn.feature_extraction.text as txt
import scipy.sparse
import GPGroups
import GPPredict
import numpy

if __name__ == "__main__":
    shp = (2,5)
    spar = scipy.sparse.lil_matrix(shp, dtype=float)
    arra = numpy.zeros(shp, dtype=float)
    arra[0,1] = 1.0
    arra[0,0] = 0.0
    arra[1,0] = 2.0
    arra[1,1] = 3.0

    #spar[0] = 2.0
    #a = spar[:,1]
    b = scipy.sparse.lil_matrix(arra[:,1])
    #a = b
    #print('type: ' + str(type(a)))
    #print(str(a.toarray()))
    #ok = spar.getrowview(1)
    #ok[0] = 90.0
    spar = spar.transpose()
    print(str(spar.toarray()))
    spar[1] = b
    spar = spar.transpose()
    print('spar:' + str(spar.toarray()))
    print('norm:' + str(arra))
    print('done')