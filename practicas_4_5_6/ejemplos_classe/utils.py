
import numpy
import datetime


def mean_absolute_error( y_true, y_pred ):
    return abs(y_true - y_pred).mean(axis=-1)

def mean_absolute_percentage_error( y_true, y_pred, epsilon=0.1 ):
    diff = abs(y_true - y_pred) / numpy.maximum( abs(y_true), epsilon )
    return 100. * diff.mean(axis=-1)

def mean_squared_error( y_true, y_pred ):
    return ((y_true - y_pred)**2).mean(axis=-1)
    

def str_to_date( s ):
    # Assuming YYYY-MM-DD
    if type(s) == str:
        return datetime.date( int(s[0:4]), int(s[5:7]), int(s[8:10] ) )
    elif type(s) == list or type(s) == numpy.ndarray:
        l=list()
        for x in s:
            l.append( str_to_date(x) )
        if type(s) == list:
            return l
        else:
            return numpy.array(l)
    else:
        raise Exception( 'Incompatible data type of input! ', type(s) )
