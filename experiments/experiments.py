#coding:utf-8

import sys
sys.path.append(sys.path[0] + '/../')
from sklearn.cluster import *
from data_loader.data_loader import DataLoader
from dist.bottom_up_edit_dist import *
from dist.vectorized_user_cate_dist import *
from ctc.density_covertree import *
from ctc.covertree_clustering import *
import time
import logging
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from config.load_config import Config

data_loader = DataLoader()
config = Config().config

def _data_format(data, precomputed=False, dist_func=None, kernal=lambda x:x):
    '''
        format data to numpy

        @data: list, each element is a data point
        @precomputed: boolean, False:not precomputed; True:precomputed
        @kernal: callable, kernal function

        #return: if precomputed => a square matrix; else => feature vec ndarray
    '''
    if not precomputed:
        return np.array(data)
    
    if dist_func is None or not callable(dist_func):
        raise Exception('a callable distance function is required')

    if kernal is not None and not callable(kernal):
        raise Exception('kernal must be callable')
    

    dist_matrix = np.array(
        [
            [ 0.0 for i in xrange(len(data))] for j in xrange(len(data))
        ]
    )
    for i in xrange(len(data)):
        for j in xrange(i, len(data)):
            if i==j:
                dist_matrix[i,j] = kernal(0.0)
            else:
                dist_matrix[i,j] = kernal(dist_func(data[i], data[j]))
    
    return dist_matrix

def rbf(dist):
    '''
        Gaussian rbf kernal function
        formula: np.exp(- (d(X,X)**2)/(2 * sigma**2 ))

        @dist: float
    '''
    #parameter modified here#
    sigma = config['rbf_sigma']

    return np.exp(-(dist**2)/(2*(sigma**2)))

def algorithm_runner(alg, dist, **kwargs):
    '''
        run algorithms; running time includes loading and converting data into acceptable format

        @alg: string, which clustering algorithm to use, in ['covertree', 'hierarichical', 'dbscan', 'kmeans', 'spectral']
        @dist: string, which distance to use, in ['vec', 'edit']
    '''

    if alg not in ['covertree', 'hierarchical', 'dbscan', 'kmeans', 'spectral']:
        raise Exception('alg in experiments not valid')
    
    if dist not in ['vec', 'edit']:
        raise Exception('dist in experiments not valid')

    #start
    start_time = time.time()

    #valid uid
    valid_uid = None if not kwargs.has_key('valid_uid') else kwargs['valid_uid']
    if valid_uid != None and type(valid_uid) != list:
        raise Exception('valid_uid must be a list')
    #data size
    data_size = float('inf') if not kwargs.has_key('data_size') else kwargs['data_size']

    #load data based on dist type
    if dist == 'vec':
        if not config.has_key('sigma'):
            raise Exception('sigma is required in vectorized distance')
        sigma = config['sigma']
        pivots = generate_category_tree(data_loader)
        data = data_loader.load(vectorized_convertor, pivots=pivots,sigma=sigma, valid_uid=valid_uid, data_size = data_size)
        metric = 'euclidean'
        X = _data_format(data, False, vectorized_dist_calculator)
    else:
        data = data_loader.load(bottomup_edit_dist_converter, valid_uid=valid_uid, data_size = data_size)
        metric = 'precomputed' 
        if alg == 'spectral':
            kernal = rbf
        else:
            kernal = lambda x:x
        X = _data_format(data, True, bottomup_edit_dist_calculator, kernal=kernal)
        
    #dbscan
    if alg == 'dbscan':
        if (not config.has_key('eps')) or (not config.has_key('min_samples')):
            raise Exception("eps and min_samples are required in config")
        eps = config['eps']
        min_samples = config['min_samples']
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(X)
        
    
    if not kwargs.has_key('k'):
        raise Exception('k is required in %s'%alg)
    k = kwargs['k']
    
    #kmeans
    if alg == 'kmeans':
        if dist == 'edit':
            raise Exception('edit distance is not supported in kmeans')
        kmeans = KMeans(n_clusters=k, max_iter=300)
        labels = kmeans.fit_predict(X)
        
    #spectral
    if alg == 'spectral':
        affinity = 'precomputed' if dist=='edit' else 'rbf'
        labels = SpectralClustering(affinity=affinity, n_clusters=k).fit_predict(X)
    
    #hierarchical
    if alg == 'hierarchical':
        labels = AgglomerativeClustering(n_clusters=k, affinity=metric, linkage='average').fit_predict(X)
    
    #covertree
    if alg == 'covertree':
        calculator = vectorized_dist_calculator if dist=='vec' else bottomup_edit_dist_calculator
        top_level = (config['edit_top_level'] if dist=='edit' else config['vec_top_level']) 
        dct = DensityCoverTree(calculator, top_level)
        for i, d in enumerate(data):
            dct.insert(Node(val=d, index=i))
        labels = covertree_clustering(dct, k)

    #end
    end_time = time.time()
    return (data, labels, end_time-start_time)

def index(data, y_predict, index_name, dist_name, y_truth = None):
    '''
        index to evaluate the experiment result

        @data: list, all data
        @y_predict: ndarray, shape(len(data),), predicted value
        @index_name: index to evaluate results, in ['sc', 'mae', 'rand']
        @dist_name: name of dist, in ['vec', 'edit']
        @y_truth: ndarray, shape(len(data),), truth
        return: float
    '''
    if index_name not in ['sc', 'mae', 'rand']:
        raise Exception('%s not supported'%index_name)
    if dist_name not in ['vec', 'edit']:
        raise Exception('%s not supported'%dist_name)

    if 'vec' == dist_name:
        dist = vectorized_dist_calculator
    else:
        dist = bottomup_edit_dist_calculator
    k = len(set(y_predict)) - (1 if -1 in y_predict else 0)
    # mae
    # here we do not calculate centers but calculate mean dist between each pair of datanodes
    # within a cluster
    if index_name == 'mae':
        clusters = [ [] for i in xrange(k) ]
        for i, cls_i in enumerate(y_predict):
            if -1 != cls_i:
                clusters[cls_i].append(data[i])
        mae = 0.0
        for clus in clusters:
            cls_mae = 0.0
            for i in xrange(len(clus)):
                for j in xrange(len(clus)):
                    cls_mae += dist(clus[i], clus[j])
            mae += cls_mae / len(clus)
        return mae
    elif index_name == 'sc':
        X = _data_format(data, precomputed=True, dist_func=dist)
        return silhouette_score(X, y_predict, metric='precomputed')
    else:
        if y_truth is None:
            raise Exception('rand index requires y_truth')
        return adjusted_rand_score(y_truth, y_predict)

def quality_experiments(dataset_name, k):
    '''
        run quality experiments, evaluate index of results and generate log

        @dataset_name: in ['testdata1000', 'randomdata1000']
    '''

    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='/log/ctclog/%s_%s_quality_exp.log'%(time.strftime("%Y-%m-%d", time.localtime()),dataset_name),
    filemode='w')

    #logging to terminal
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    algs = ['covertree', 'kmeans', 'spectral', 'hierarchical'] #spectral todo
    dists = ['vec', 'edit']
    indexs = ['sc', 'mae', 'rand']
    with open(dataset_name,'r') as valid_uid_f:
            valid_uid = valid_uid_f.read().split('\n')

    y_truth = None
    if dataset_name == 'testdata1000':
        with open('testtruth','r') as truth_f:
            y_truth = truth_f.read().split('\n')
        for i in xrange(len(y_truth)):
            y_truth[i] = int(y_truth[i])
    
    for alg in algs:
        for dist in dists:
            if alg=='kmeans' and dist=='edit':
                continue
            data, labels, run_time = algorithm_runner(alg, dist, valid_uid=valid_uid, k=k)
            log_content = 'k:%s; dataset:%s; alg:%s; distance_type:%s; runtime:%d; ' % (k, dataset_name, alg, dist, run_time)
            #index
            for idx in indexs:
                index_val = index(data, labels, idx, dist, y_truth)
                log_content += '%s:%s; '%(idx, str(index_val))
            #size of clusters
            log_content += 'size: [ '
            for c in set(labels):
                log_content += '%d '%labels.tolist().count(c)
            log_content += ']'

            logging.info(log_content)
        

def efficiency_experiments(data_size):
    '''
        run efficiency experiments, evaluate index of results and generate log

        @dataset_size: number
    '''
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='/log/ctclog/%s_effi_exp.log'%(time.strftime("%Y-%m-%d", time.localtime())),
    filemode='w')

    #logging to terminal
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    algs = ['covertree']
    dists = ['vec']

    for alg in algs:
        for dist in dists:
            if alg=='kmeans' and (dist=='edit' or data_size > 25000):
                continue
            if alg=='spectral' and data_size > 5000:
                continue
            if alg=='hierarchical' and data_size > 50000000:
                continue
            if alg=='kmeans' and data_size > 25000:
                continue
            if alg=='dbscan' and data_size > 25000:
                continue
            data, labels, run_time = algorithm_runner(alg, dist, data_size=data_size, k=20)
            log_content = 'k:%s; data_size:%d; alg:%s; distance_type:%s; runtime:%d; ' % (20, data_size, alg, dist, run_time)

            logging.info(log_content)


if len(sys.argv) == 1:
    raise Exception('which experiments?')
elif sys.argv[1] == 'quality':
    for dataset in ['testdata1000','randomdata1000']:
        for k in xrange(2, 20):
            quality_experiments(dataset, k)
elif sys.argv[1] == 'efficiency':
    for data_size in [500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 25000, 40000, 80000, 100000, 200000, 250000,300000, 350000, 400000, 450000, 500000, 600000, 800000, 1000000, 1200000]:
        efficiency_experiments(data_size)

# efficiency_experiments(10000)


