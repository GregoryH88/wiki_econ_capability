import pywikibot
from pywikibot import pagegenerators
import mwparserfromhell as pfh
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import scipy.stats as ss
from scipy.optimize import fmin as scipyfmin
import operator
import re
import json
from matplotlib.mlab import PCA
import shelve
import os


def load_files(folder):
    if not folder.endswith('/'):
        folder = folder+'/'
    M = np.load(folder+'M.npy')
    user_dict = json.load(open(folder+'user_dict.json', 'r'))
    article_dict = json.load(open(folder+'article_dict.json', 'r')) 
    user_exogenous_ranks = json.load(open(folder+'user_exogenous_ranks.json', 'r'))
    article_exogenous_ranks = json.load(open(folder+'article_exogenous_ranks.json', 'r'))
    return {'M':M,
            'user_dict':user_dict,'article_dict':article_dict,
            'user_exogenous_ranks':user_exogenous_ranks, 'article_exogenous_ranks':article_exogenous_ranks}
    
def make_bin_matrix(M):
    #this returns the a matrix with entry True where the original was nonzero, and zero otherwise.
    M[M>0] = 1.0
    return M

def M_test_triangular(M):
    user_edits_sum = M.sum(axis=1)
    article_edits_sum = M.sum(axis=0)
    
    user_edits_order = user_edits_sum.argsort()
    article_edits_order = article_edits_sum.argsort()
    
    M_sorted = M[user_edits_order,:]
    M_sorted_sorted = M_sorted[:,article_edits_order]
    
    M_bin = make_bin_matrix(M_sorted_sorted)

def w_star_analytic(M, alpha, beta, w_star_type):
    k_c  = M.sum(axis=1) #aka k_c summing over the rows
    k_p = M.sum(axis=0) #aka k_p summering over the columns
    
    A = 1
    B = 1
    
    def Gcp_denominateur(M, p, k_c, beta):
        M_p = M[:,p]
        k_c_beta = k_c ** (-1 * beta)
        return np.dot(M_p, k_c_beta)
    
    def Gpc_denominateur(M, c, k_p, alpha):
        M_c = M[c,:]
        k_p_alpha = k_p ** (-1 * alpha)
        return np.dot(M_c, k_p_alpha)
    
    if w_star_type == 'w_star_c':
        w_star_c = np.zeros(shape=M.shape[0])

        for c in range(M.shape[0]):
            summand = Gpc_denominateur(M, c, k_p, alpha)
            k_beta = (k_c[c] ** (-1 * beta))
            w_star_c[c] = A * summand * k_beta

        return w_star_c
    
    elif w_star_type == 'w_star_p':
        w_star_p = np.zeros(shape=M.shape[1])
    
        for p in range(M.shape[1]):
            summand = Gcp_denominateur(M, p, k_c, beta)
            k_alpha = (k_p[p] ** (-1 * alpha))
            w_star_p[p] = B * summand * k_alpha
    
        return w_star_p
    
def Gcp_denominateur(M, p, k_c, beta):
    M_p = M[:,p]
    k_c_beta = k_c ** (-1 * beta)
    return np.dot(M_p, k_c_beta)

def Gpc_denominateur(M, c, k_p, alpha):
    M_c = M[c,:]
    k_p_alpha = k_p ** (-1 * alpha)
    return np.dot(M_c, k_p_alpha)


def make_G_hat(M, alpha=1, beta=1):
    '''G hat is Markov chain of length 2
    Gcp is a matrix to go from  contries to product and then 
    Gpc is a matrix to go from products to ccountries'''
    
    k_c  = M.sum(axis=1) #aka k_c summing over the rows
    k_p = M.sum(axis=0) #aka k_p summering over the columns
    
    G_cp = np.zeros(shape=M.shape)
    #Gcp_beta
    for [c, p], val in np.ndenumerate(M):
        numerateur = (M[c,p]) * (k_c[c] ** ((-1) * beta))
        denominateur = Gcp_denominateur(M, p, k_c, beta)
        G_cp[c,p] = numerateur / float(denominateur)
    
    
    G_pc = np.zeros(shape=M.T.shape)
    #Gpc_alpha
    for [p, c], val in np.ndenumerate(M.T):
        numerateur = (M.T[p,c]) * (k_p[p] ** ((-1) * alpha))
        denominateur = Gpc_denominateur(M, c, k_p, alpha)
        G_pc[p,c] = numerateur / float(denominateur)
    
    
    return {'G_cp': G_cp, "G_pc" : G_pc}

def w_generator(M, alpha, beta):
    #this cannot return the zeroeth iteration
    
    G_hat = make_G_hat(M, alpha, beta)
    G_cp = G_hat['G_cp']
    G_pc = G_hat['G_pc']
    #

    fitness_0  = np.sum(M,1)
    ubiquity_0 = np.sum(M,0)
    
    fitness_next = fitness_0
    ubiquity_next = ubiquity_0
    i = 0
    
    while True:
        
        fitness_prev = fitness_next
        ubiquity_prev = ubiquity_next
        i += 1
        
        fitness_next = np.sum( G_cp*ubiquity_prev, axis=1 )
        ubiquity_next = np.sum( G_pc* fitness_prev, axis=1 )
        
        yield {'iteration':i, 'fitness': fitness_next, 'ubiquity': ubiquity_next}
        


def w_stream(M, i, alpha, beta):
    """gets the i'th iteration of reflections of M, 
    but in a memory safe way so we can calculate many generations"""
    if i < 0:
        raise ValueError
    for j in w_generator(M, alpha, beta):
        if j[0] == i:
            return {'fitness': j[1], 'ubiquity': j[2]}
            break
            
def find_convergence(generator, M, alpha, beta, fit_or_ubiq, do_plot=False,):
    '''finds the convergence point (or gives up after 1000 iterations)'''
    if fit_or_ubiq == 'fitness':
        Mshape = M.shape[0]
    elif fit_or_ubiq == 'ubiquity':
        Mshape = M.shape[1]
    
    rankings = list()
    
    prev_rankdata = np.zeros(Mshape)
    iteration = 0
    
    for stream_data in generator(M, alpha, beta):
        iteration = stream_data['iteration']
        
        data = stream_data[fit_or_ubiq]
        rankdata = data.argsort().argsort()
        
        #test for convergence
        if np.equal(rankdata,prev_rankdata).all():
            break
        if iteration == 1000:
            break
        else:
            rankings.append(rankdata)
            prev_rankdata = rankdata


def rank_comparison(a_ranks_sorted, b_ranks_sorted, do_plot=False):
    a_list = list()
    b_list = list()
    for atup in a_ranks_sorted:
        aiden = atup[0]
        apos = atup[1]
        #find this in our other list
        for btup in b_ranks_sorted:
            biden = btup[0]
            bpos = btup[1]
            if aiden == biden:
                a_list.append(apos)
                b_list.append(bpos)

    
    return ss.spearmanr(a_list, b_list)


"""
the evolution of rho and alpha and beta as  which iteration step we stop on
we will optimize later. for now, we have a combinatorial problem 
and we are going to np-completely enumerate over all alpha, beta, and w_iterations

"""
#this is how we'll sort the scroes and add to our dataframe
etagere = pd.DataFrame(columns=['cat','alpha','beta','iteration','rho','user_or_article'])

def compare_and_add(iteration, rank, user_or_article, user_or_art_dict, exo_ranks_sorted, alpha, beta):
        endo_ranks = {name: rank[pos] for name, pos in user_or_art_dict.iteritems()}
        endo_ranks_sorted = sorted(endo_ranks.iteritems(), key=operator.itemgetter(1))

        spearman = rank_comparison(endo_ranks_sorted, exo_ranks_sorted) 

        #check significance
        if spearman[1] < 0.05:
            rho = spearman[0]
        else:
            #will just have to remember that zero-means not significant, 
            #doubtful that 0, the int could actually be produced
            rho = 0
        global etagere
        etagere = etagere.append({'cat':cat_name,
                        'alpha':alpha,
                        'beta':beta,
                        'iteration':iteration,
                        'rho':rho,
                        'user_or_article':user_or_article}, ignore_index=True)

savedata = '/home/notconfusing/workspace/contagion/savedata'
for cat_name in os.listdir(savedata):
    if cat_name == 'Category:Feminist_writers':
        print cat_name
        dates_dir = os.path.join(savedata, cat_name)
        dates = os.listdir(dates_dir)
        latest = max(dates)
        taken_date_path = os.path.join(dates_dir, latest)
        #print 'datepath',taken_date_path
        snapshot_dates = os.listdir(taken_date_path)
        #print snapshot_dates
        latest_snapshot = max(snapshot_dates)
        latest_snapshot_dir = os.path.join(taken_date_path, latest_snapshot)
        file_group = load_files(latest_snapshot_dir)

        #per-category vars
        M = make_bin_matrix(file_group['M'])
        user_dict = file_group['user_dict']
        article_dict = file_group['article_dict']
        exo_user_ranks = file_group['user_exogenous_ranks']
        exo_article_ranks = file_group['article_exogenous_ranks']

        #this is the landscape size we are sampling
        square_bound = 2
        resolution = 2

        for alpha in np.arange(start=(-1*square_bound), stop=(square_bound + resolution), step=resolution):
            for beta in np.arange(start=(-1*square_bound), stop=(square_bound + resolution), step=resolution):
                print('alpha, beta', alpha, beta)
                
                class iteration_state():
                    def __init__(self, user_or_article, prev_rank):
                        self.user_or_article = user_or_article
                        self.prev_rank = None
                        self.converged = False
                        if user_or_article == 'user':
                            self.fit_or_ubiq = 'fitness'
                            self.user_or_art_dict = user_dict
                            self.exo_ranks_sorted = exo_user_ranks
                        else:
                            self.fit_or_ubiq = 'ubiquity'
                            self.user_or_art_dict = article_dict
                            self.exo_ranks_sorted = exo_article_ranks
                            
                #seed 0th iteration
                #and initial conditions
                iteration = 0
                
                prev_user_rank = np.sum(M, axis=1).argsort().argsort()
                prev_article_rank = np.sum(M, axis=0).argsort().argsort()
                
                user_state = iteration_state('user', prev_user_rank)
                article_state = iteration_state('article', prev_article_rank)


                #start streaming data!
                def start_streaming():
                    stream = w_generator(M, alpha, beta)
                    for stream_data in stream:
                        for user_or_article in ['user','article']:
                            #load the right flags to use for either user or article
                            #excuse the ternary operator making this a one liner, 
                            #I am trying to clear some space to the rest is readable 
                            #and remove redundant code
                            if user_or_article == 'user':
                                state = user_state
                            else:
                                state = article_state
  
                            iteration = stream_data['iteration']
    
                            data = stream_data[state.fit_or_ubiq]
                            rank = data.argsort().argsort()
    
                            #let someone else compare and store the data
                            compare_and_add(iteration = iteration,
                                            rank = rank,
                                            user_or_article = state.user_or_article,
                                            user_or_art_dict = state.user_or_art_dict,
                                            exo_ranks_sorted = state.exo_ranks_sorted,
                                            alpha = alpha,
                                            beta = beta)
    
                            elem_equal = np.equal(rank, state.prev_rank)
                            print('iteration', iteration, 'ua', state.user_or_article, '%equal', sum([int(boole) for boole in elem_equal]) / float(len(elem_equal)))
                            if np.equal(rank, state.prev_rank).all():
                                state.converged = True
                                
                            #remember the last state to check against
                            state.prev_rank = rank
    
                            #test for convergence
                            if user_state.converged and article_state.converged:
                                    print('converged at iteration: ', iteration)
                                    return
                            #test for probable never convergence
                            if iteration >= 100:
                                print('Max iterations reached')
                                return
                    
                start_streaming()


    '''
    for user_or_articles in ['users','articles']:
        if user_or_articles == 'users':
            exo_rank_file, name_dict, initial_guess = 'user_exogenous_ranks', 'user_dict', [0,-2]
        else:
            exo_rank_key, name_dict, initial_guess = 'article_exogenous_ranks', 'article_dict', [-2,0]
    '''
print etagere
etagere.to_pickle(os.path.join(savedata,'etagere'))