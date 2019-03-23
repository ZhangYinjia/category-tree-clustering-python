#coding: utf-8

import sys
sys.path.append(sys.path[0] + '/../')
from data_loader.data_loader import DataLoader
import numpy as np

class CateTreeNode:

    def __init__(self, label, parent):
        '''
            init function of CateTreeNode

            @laebl: label of the node
            @parent: parent node
        '''
        if type(label)!=str and type(label)!=unicode:
            raise Exception('label must be str or unicode')
        if parent is not None and parent.__class__!=CateTreeNode:
            raise Exception('parent must be a CateTreeNode')

        self.label = label
        self.parent = parent
        self.bus_cnt = 0 #count of business whose tail category is represented by this node
        self.chd_set = [] #children set
    
    def find_label_in_chd(self, label):
        '''
            find if there exists a node in chd_set whose label is same with @label

            @label: label to find

            #return: None if not exists, else the instance
        '''
        if type(label)!=str and type(label)!=unicode:
            raise Exception('label must be str or unicode')
        
        for n in self.chd_set:
            if n.label == label:
                return n
        return None
    
class CateTree:

    def __init__(self):
        '''
            init function of CateTree
        '''
        self.root = CateTreeNode(label='', parent=None)
    
    def insert(self, cate_path):
        '''
            insert a category path to category tree, if the path has been
            inserted, then add 1 to bus_cnt
            
            @cate_path: a category path 
        '''

        if type(cate_path) != list:
            raise Exception('cate_path must be a list!')

        current_node = self.root
        for i in xrange(len(cate_path)):
            tmp_label = cate_path[i]
            tmp_chd_set = current_node.chd_set
            
            next_node = None
            for n in tmp_chd_set:
                if n.label == tmp_label:
                    next_node = n
                    break
            
            if next_node is None:
                next_node = CateTreeNode(label=tmp_label, parent=current_node)
                current_node.chd_set.append(next_node)
            current_node = next_node
        current_node.bus_cnt += 1
    
    def similarity(self, path_set):
        '''
            calculate similarity between a user and this tree

            @path_set: a set of paths, [ [<path_1>], [<path_2>], ...]

            #return: a float
        '''

        if type(path_set)!=list:
            raise Exception('path_set must be list')

        sum_similarity = 0.0
        for path in path_set:
            similarity = 1.0
            found = True
            p = self.root
            for i in xrange(len(path)):
                label = path[i]
                user_cate_node = p.find_label_in_chd(label)
                if user_cate_node is not None:
                    bus_share = 0.0 if user_cate_node.bus_cnt==0 else 1.0
                    similarity *= 1.0/(len(user_cate_node.chd_set) + bus_share)
                    if i == len(path)-1:
                        similarity *= 1.0/user_cate_node.bus_cnt
                    
                    p = user_cate_node
                else:
                    found = False
                    break
            
            if found:
                sum_similarity += similarity
      
        return sum_similarity


def vectorized_convertor(uid, bus_cate_dict, kwargs):
    '''
        convert a user's category data to data a vector

        @uid: user id
        @bus_cate_dict: a dict whose keys are business ids and values are category paths
        @kwargs: a dict of other parameter, {'pivots':[], 'sigma': val} 

        #return: feature vector
    '''
    
    if type(bus_cate_dict) != dict:
        raise Exception('bus_cate_dict must be a dict')

    pivots = kwargs['pivots']
    if type(pivots) != list:
        raise Exception('pivots in kwargs must be a list of CateTree')
    try:
        sigma = kwargs['sigma']
    except Exception, e:
        sigma = 0.
    if type(sigma) != float and type(sigma)!=int and type(sigma)!=list:
        raise Exception('sigma in kwargs must be a float or an int or an list')
    if kwargs.has_key("mean"):
        mean = kwargs["mean"]
    else:
        mean = None

    feature_arr = [0.0 for i in xrange(len(pivots))]
    path_sets = []
    for key in bus_cate_dict.keys():
        for p in bus_cate_dict[key]:
            path_sets.append(p)
    for d in xrange(len(pivots)):
        t = pivots[d]
        if t.__class__ != CateTree:
            raise Exception('the %d-th pivot not a CateTree' % d)
        feature_arr[d] = t.similarity(path_sets)
        if sigma != 0.:
            if type(sigma) == list:
                s = sigma[d]
            else:
                s = sigma
            u = t.similarity(path_sets)
            feature_arr[d] = np.exp(-np.power(u, 2)/(2.0*np.power(s, 2)))

    feature_arr = np.array(feature_arr)
    return feature_arr

def vectorized_dist_calculator(v_1, v_2):
    '''
        calculate distance between v_1 and v_2
        @v_1:
        @v_2:

        #return: distance value, a float
    '''
    if type(v_1)!=np.ndarray or type(v_2)!=np.ndarray:
        raise Exception('parameters must be np.ndarray')
    
    if np.shape(v_1) != np.shape(v_2):
        raise Exception('v_1 and v_2 must share a common shape')

    return np.sqrt(np.sum(np.square(v_1-v_2)))


def generate_category_tree(data_loader):
    '''
        generate CateTree pivots 

        @data_loader: an instance of DataLoader

        #return: a list of CateTree
    '''
    if data_loader.__class__ != DataLoader:
        raise Exception('data_loader must be an instance of DataLoader')

    pivots_dict = {}
    all_paths = data_loader.get_all_cate_path()
    for path in all_paths:
        tree_name = path[0]
        if not pivots_dict.has_key(tree_name):
            pivots_dict[tree_name] = CateTree()
        pivots_dict[tree_name].insert(path)
    
    return pivots_dict.values()

    
