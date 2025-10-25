

from pyspark import SparkContext
import findspark
findspark.init()

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import time


def run_training(workers = 4, partitions=None, cacheMode=None):


    sc = SparkContext(master=f"local[{workers}]", appName="TextFileExample")

    DATA_FILE = "botnet_tot_syn_l_debug.csv"
    X_SIZE = 11
    Y_SIZE = 1
    N_ITER = 10
    LEARNING_RATE = 1.5


    def readFile(filename):
        if cacheMode=="source" or cacheMode=="opt":
            rdd = sc.textFile(filename).cache()
        else:
            rdd = sc.textFile(filename)
        def map_line(line):
            elements = [float(element) for element in line.split(",")]
            return (np.array(elements[:-1]), int(elements[-1]))
        return rdd.map(map_line)


    def normalize (RDD_Xy):
        if cacheMode=="map":
            rdd_col = RDD_Xy.map(lambda xy: (np.array(xy[:-1], dtype=float).flatten(), int(xy[-1]))).cache()
            sum_vec = (rdd_col.map(lambda xy: xy[0]).cache()).reduce(lambda a, b: a + b)
        elif cacheMode=="opt":
            rdd_col = RDD_Xy.map(lambda xy: (np.array(xy[:-1], dtype=float).flatten(), int(xy[-1]))).cache()
            sum_vec = rdd_col.map(lambda xy: xy[0]).reduce(lambda a, b: a + b)
        else:
            rdd_col = RDD_Xy.map(lambda xy: (np.array(xy[:-1], dtype=float).flatten(), int(xy[-1])))
            sum_vec = rdd_col.map(lambda xy: xy[0]).reduce(lambda a, b: a + b)
        media = sum_vec / n
        if cacheMode=="map":
            varianza = (rdd_col.map(lambda v: (v[0]-media)**2).cache()).reduce(lambda a,b:a+b)/n
        else:
            varianza = rdd_col.map(lambda v: (v[0]-media)**2).reduce(lambda a,b:a+b)/n
        std=np.sqrt(varianza)
        if cacheMode=="map" or cacheMode=="opt":
            norm = rdd_col.map(lambda v: ((v[0] - media)/std, v[1])).cache()
        else:
            norm = rdd_col.map(lambda v: ((v[0] - media)/std, v[1]))
        return norm

    def train(RDD_Xy, iterations, learning_rate):
        sigma = lambda z : (1 / (1 + (np.e**(-z))))
        W = np.array([np.random.normal(0, 1) for _ in range(X_SIZE)])
        b = 0
        def calculate_dw(rdd, W, b):
            if cacheMode=="map":
                rdd = rdd.map(lambda xy: np.array([(sigma(np.dot(W, xy[0]) + b) - xy[1]) * x_i for x_i in xy[0]])).cache()
            else:
                rdd = rdd.map(lambda xy: np.array([(sigma(np.dot(W, xy[0]) + b) - xy[1]) * x_i for x_i in xy[0]]))
            dw = rdd.reduce(lambda a, b : a + b) / n           
            return dw
        
        def calculate_db(rdd, W, b):
            if cacheMode=="map":
                rdd = rdd.map(lambda xy: sigma(np.dot(W, xy[0]) + b) - xy[1]).cache()
            else:
                rdd = rdd.map(lambda xy: sigma(np.dot(W, xy[0]) + b) - xy[1])
            db = rdd.reduce(lambda a, b : a + b) / n        
            db = rdd.reduce(lambda a, b : a + b) / n
            return db
        
        for _ in range(iterations):
            dw = calculate_dw(RDD_Xy, W, b)
            db = calculate_db(RDD_Xy, W, b)
            W = W - learning_rate * dw
            b = b - learning_rate * db
        
        return W, b

    def accuracy(w, b, RDD_Xy):
        if cacheMode=="map":
            predictions = RDD_Xy.map(lambda v: predict(w, b, v[0])).cache()
        else:
            predictions = RDD_Xy.map(lambda v: predict(w, b, v[0]))
        count = predictions.reduce(lambda a, b : a + b)
        return count/n

    def predict(w, b, data):
        sigma = lambda z : (1 / (1 + (np.e**(-z))))
        res = sigma((w*data[0]).sum() + b)
        if(res>=0.5): 
            return 1
        else:
            return 0



    data = readFile(DATA_FILE)
    n = data.count()
    
    start_time = time.time()
    
    rdd_norm=normalize(data)
    
    if partitions != None:
        rdd_norm = rdd_norm.repartition(partitions)
    
    W, b = train(rdd_norm, N_ITER, LEARNING_RATE)
    acc = accuracy(W, b, rdd_norm)

    sc.stop()
    
    execution_time = time.time() - start_time
    
    
    return execution_time, acc
    


if __name__ == "__main__":
    print(run_training(3, 6))


