
# coding: utf-8

# #Method of reflections in python
# The Method of Reflection (MOR) is a algorithm first coming out of Macroeconomics, that ranks nodes in a bi-partite network. This notebook should hopefully help you implement the _method of reflection_ in python. To be precise, it is the modified algorithm that is proposed by Caldarelli et al., which solves some problems with the original Hidalgo-Hausmann (HH) algorithm [doi:10.1073/pnas.0900943106](http://chidalgo.com/Papers/HidalgoHausmann_PNAS_2009_PaperAndSM.pdf). The main problem with (HH) is that all values converge to a single fixed point after sufficiently many iterations. The Caldarelli version solves this by adding a new term to the recursive equation - what they call a _biased random walker_ (function _G_). [doi: 10.1371/journal.pone.0047278](http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0047278). I hadn't seen any open-source implementations of this algorithm, so I thought I'd share my naïve approach.

# In[4]:

import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as ss
from scipy.optimize import fmin as scipyfmin
import operator
import re
import json

#if you are using ipython and want to see things inline. 
get_ipython().magic(u'pylab inline')


# #Modelling the data
# We want to model a bi-partite network. Since we are using `numpy` and want to operate on a matrix we will use a `numpy.matrix`, but we also may want to retain the Unique IDs associated with each node, so we'll need to keep `dicts` of those as they relate to the matrix indices. Therefore for each bi-partite network at least __three files__ are needed. In my case I analysed a network which was a Wikipedia Category, so my two node-types are _users_ and _articles_. In your case they may be different, but I've kept the nomenclature here for simplicity - my own simplicity 8^). You can borrow my sample data from https://github.com/notconfusing/wiki_econ_capability/tree/master/savedata
# 
# + the `M.npy` - an numpy matrix _M_ - adjacently matrix of the network.
# + `user_dict.json` - a mapping (json dictionary)between unique key (in my case Wikipedia user name) to _M_ row index.
# + `article_dict` - a mapping (json dictionary) between unique key (in my case the name of the article) _M_ column index.
# 
# ##Extra data
# If you plan to calibrate your model against some exogenous metrics you will need to provide two more files - the rankings from the exogenous data.
# 
# + `user_exogenous_ranks.json` - a mapping (json dictionary) between the same keys of the  `user_dict` to their exogenous ranks (how well another metric ranks Wikipedia users).
# + `article_exogenous_ranks.json` - - a mapping (json dictionary) between the same keys of the  `article_dict` to their exogenous ranks (how well another metric ranks Wikipedia articles).
# 
# ##Putting it all in a folder
# I put everything belonging to one network into a single folder and then use this `load_files` method which unpacks all the object and gives you a dict of all the objects.

# In[2]:

def load_files(folder):
    M = np.load(folder+'M.npy')
    user_dict = json.load(open(folder+'user_dict.json', 'r'))
    article_dict = json.load(open(folder+'article_dict.json', 'r')) 
    user_exogenous_ranks = json.load(open(folder+'user_exogenous_ranks.json', 'r'))
    article_exogenous_ranks = json.load(open(folder+'article_exogenous_ranks.json', 'r'))
    return {'M':M,
            'user_dict':user_dict,
            'article_dict':article_dict,
            'user_exogenous_ranks':user_exogenous_ranks, 
            'article_exogenous_ranks':article_exogenous_ranks}


# For instance, let's create the `feminist_data` dict, which holds data from _Category:Feminist_writers_ . I am snapshotting data, so that's why below you see two dates: when the snapshot was taken, and the latest date to be considered in the snapshot. So we are getting the full data from February 18th 2014.

# In[7]:

feminist_data = load_files('savedata/Category:Feminist_writers/2014-02-18/2014-02-18/')


# Now we define two operations on _M_. The _HH_ technique and those coming after use a binary input matrix, I also got better results, with a binary matrix. So we can normalise all our non-zero data to `1`. Also we may want to take a look at it an see if it has a triangular shape, since (MOR) assumes a triangular matrix.

# In[8]:

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
    
    plt.figure(figsize=(10,10))
    imshow(M_sorted_sorted, cmap=plt.cm.bone, interpolation='nearest')


# In[10]:

bin_M = make_bin_matrix(feminist_data['M'])
M_test_triangular(bin_M)


# Ok now let's move onto our third and fourth operations on _M_ - getting some numeric rankings out of the matrix. Let us refresh ourselves on the equations from the literature. (Note variables _c_ and _p_, countries and products, translate to editors and article respectively): 
# 
# ##Zeroth order scores
# These are an $w_{c}$ editor-vector which is the sums of articles edited by each editor. Or the article-vector $w_{p}$, which is the sum of editors contributing to each article.
# 
# \begin{cases}
#  w_{c}^{(0)} = \sum_{p=1}^{N_{p}} M \equiv k_c\\[7pt]
#  w_{p}^{(0)} = \sum_{c=1}^{N_{c}} M \equiv k_p
# \end{cases}
# 
# ## Higher orders
# The first order $w^{1}_c$ is the sum of the articles touched, but weighted by the Zeroth order article-vector (and the $G$ term). So if you've edited better articles that counts. And $w^{1}_c$ is the sum of editors touching, but weighted by the Zeroth order editor-vector (and $G$). So if you're touched by better editors that's also being considered. 
# 
# Beyond the first order interpretation for the higher orders is difficult.
# 
# \begin{cases}
# w^{(n+1)}_c (\alpha,\beta) = \sum_{p=1}^{N_p}  G_{cp}(\beta) \,w^{(n)}_p (\alpha,\beta)\\[7pt]
# w^{(n+1)}_p (\alpha,\beta) = \sum_{c=1}^{N_c}  G_{pc}(\alpha) \, w^{(n)}_c (\alpha,\beta)\\
# \end{cases}
# 
# ## G - transition probability function
# Depending on $\alpha$ and $\beta$ we non-linearly weight based on the Zeroth order iterations. 
# 
# \begin{cases}
# G_{cp}(\beta) = \frac{M_{cp} k_{c}^{-\beta}}{\sum_{c' = 1}^{N_c} M_{c'p} k_{c'}^{-\beta}}\\[10pt]
# G_{pc}(\alpha) = \frac{M_{cp} k_{p}^{-\alpha}}{\sum_{p' = 1}^{N_p} M_{cp'} k_{p'}^{-\alpha}}.\\
#  \end{cases}
# 
# ## Translating the mathematics into numpy
# And now we implement the mathematics in python. Hopefully I got this right, it hasn't been independently verified. Additionally I implement $w$ as a `generator`, so you can go on for many generations without chewing up too much memory. There is also a stream function that allows you get a specific iteration. And lastly a `find_convergence` function, that checks to see if the rankings haven't shifted for two consecutive iterations.

# In[71]:

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
            
def find_convergence(M, alpha, beta, fit_or_ubiq, do_plot=False,):
    '''finds the convergence point (or gives up after 1000 iterations)'''
    if fit_or_ubiq == 'fitness':
        Mshape = M.shape[0]
    elif fit_or_ubiq == 'ubiquity':
        Mshape = M.shape[1]
    
    rankings = list()
    scores = list()
    
    prev_rankdata = np.zeros(Mshape)
    iteration = 0

    for stream_data in w_generator(M, alpha, beta):
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
            scores.append(data)
            prev_rankdata = rankdata
            
    if do_plot:
        plt.figure(figsize=(iteration/10, Mshape / 20))
        plt.xlabel('Iteration')
        plt.ylabel('Rank, higher is better')
        plt.title('Rank Evolution')
        p = semilogx(range(1,iteration), rankings, '-,', alpha=0.5)
    return {fit_or_ubiq:scores[-1], 'iteration':iteration}


# We also know from Caldarelli et al. that there is an analytic formulation to the recursive procedure. So if you want to save some (a lot) processing and just know the end result we can use:
# 
# ## Analytic solution
# 
# \begin{cases}
# w^{*}_e (\alpha,\beta) = (\sum_{a=1}^{N_a} M_{ea}k_{a}^{-\alpha})k_{e}^{-\beta} \\
# w^{*}_a (\alpha,\beta) = (\sum_{e=1}^{N_e}  M_{ea}k_{e}^{-\beta})k_{a}^{-\alpha}\\
# \end{cases}
# 
# And again in python:

# In[13]:

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


# ## Running the iterative and analytic solutions on data.
# We will run our algorithms on our data. The output of both the iterative and the analytic solutions are scores. So in order to know who was the best, we afterwards identify (this is why we need the ID-mapping) and sort the list. I use `pandas` here to simply life, but I've also done it in pure-python if you're not familiar with `pandas`. Also I arbitrarily use $(\alpha, \beta) = (0,0)$.
# 
# ###First lets use the analytic solution.

# In[25]:

#purer python
#score
w_scores = w_star_analytic(M=feminist_data['M'], alpha=0.5, beta=0.5, w_star_type='w_star_c')
#identify
w_ranks = {name: w_scores[pos] for name, pos in feminist_data['user_dict'].iteritems() }
#sort
w_ranks_sorted = sorted(w_ranks.iteritems(), key=operator.itemgetter(1))

#or use pandas
w_scores_df = pd.DataFrame.from_dict(w_ranks, orient='index')
w_scores_df.columns = ['w_score']
w_scores_df.sort(columns=['w_score'], ascending=False).head()


# Well done users _Dsp13_ and _Bearcat_. (Isn't there a Mogwai video called _Bearcat_? Oh, no it's [Batcat](https://www.youtube.com/watch?v=KMDCM5OAOaE) \\m/ never mind let's move on.) 
# 
# ##Verification with the iterative method.
# Let's take the long way home, and check that the shortcut actually takes us to the right place. We use the iterative method with the same data, until we find convergence. Also I make a plot here that ranks the users after each iteration, so we can track them. So each line you will see if the history of users rise to Glory, or their slow decline to forgotten irrelevance (or none of those phenomenon). Actually if you see a user going up its because the value of the articles edited is increasing. And likewise if a user is losing standing, its because they edited a lot of articles, but were of poor quality.

# In[ ]:

convergence = find_convergence(M=feminist_data['M'], alpha=0.5, beta=0.5, fit_or_ubiq='fitness', do_plot=True)


# In[41]:

convergence['iteration']


# It looks like it took 661 iterations for this particular network to converge with $(\alpha, \beta) = (0,0)$. Now, let's what the ranking produce. 

# In[47]:

w_iter = convergence['fitness'] 
#rank
w_iter_ranks = {name: w_iter[pos] for name, pos in feminist_data['user_dict'].iteritems() }
#sort
w_ranks_sorted = sorted(w_iter_ranks.iteritems(), key=operator.itemgetter(1))

#or use pandas
w_iter_scores_df = pd.DataFrame.from_dict(w_iter_ranks, orient='index')
w_iter_scores_df.columns = ['w_score']
w_iter_scores_df.sort(columns=['w_score'], ascending=False).head()


# Verified. You can see the analytic and iterative methods produce the same ranking, but the scores are off by a normalisation constant.
# 
# ## Calibration with exogenous variables.
# 
# Lastly, we might want to know for what values of $alpha$ and $beta$ maximise our correlation between model and actual. Without resorting to any fancy optimisers, let's just perform a grid search. With a gridserch we can also get a picture of the landscape. We define a way to compare our list rankings using the Spearman method from `scipy.stats`. Then we'll make a landscape of $[-2,2] \times [-2,2]$ with resolution $50 \times 50$, and evaluate at all those points (using the analytic solution). Finally we will return the top correlation we found.

# In[67]:

'''I'm sure this can be done much more elegantly
but this was sort-of drink-a-lot-of-coffee-one-afternoon-and-get-it-done
cleaning this up is an exercise for the reader'''
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
    if do_plot:    
        plt.figure(figsize=(10,20))
        plot([1,2], [a_list, b_list], '-o')
        plt.show()
    
    return ss.spearmanr(a_list, b_list)

def calibrate_analytic(M, ua, exogenous_ranks_sorted, user_or_art_dict, index_function, title, do_plot=False):
    
    if ua == 'users':
        w_star_type = 'w_star_c'
    elif ua == 'articles':
        w_star_type = 'w_star_p'
    
    squarelen = range(0,50)
    
    alpha_range = map(index_function,squarelen)
    beta_range = map(index_function,squarelen)
    landscape = np.zeros(shape=(len(list(alpha_range)),len(list(beta_range))))

    top_spearman = {'spearman':None,'alpha':None, 'beta':None, 'ua':ua}

    for alpha_index, alpha in enumerate(alpha_range):
        for beta_index, beta in enumerate(beta_range):
            
            w_converged = w_star_analytic(M, alpha, beta, w_star_type)
            
            w_ranks = {name: w_converged[pos] for name, pos in user_or_art_dict.iteritems() }
            w_ranks_sorted = sorted(w_ranks.iteritems(), key=operator.itemgetter(1))
            
            spearman = rank_comparison(w_ranks_sorted, exogenous_ranks_sorted)

            if spearman[1] < 0.05:
                landscape[alpha_index][beta_index] = spearman[0]
                
                if (not top_spearman['spearman']) or (spearman[0] > top_spearman['spearman']):
                    top_spearman['spearman'] = spearman[0]
                    top_spearman['alpha'] = alpha
                    top_spearman['beta'] = beta
            else:
                landscape[alpha_index][beta_index] = np.nan

    if do_plot:
        plt.figure(figsize=(10,10))
        heatmap = imshow(landscape, interpolation='nearest', vmin=-1, vmax=1)
        #heatmap = plt.pcolor(landscape)
        colorbar = plt.colorbar(heatmap)
        plt.xlabel(r'$ \beta $')
        plt.xticks(squarelen, beta_range, rotation=90)
        plt.ylabel(r'$ \alpha $')
        plt.yticks(squarelen, alpha_range)
        plt.title(title)
        
        landscape_file = open(title+'_landscape.npy', 'w')
        np.save(landscape_file, landscape)
        plt.savefig(title+'_landscape.eps')

    return top_spearman


# Ok, now let's run the calibration and get our optimising variables.

# In[ ]:

user_spearman = calibrate_analytic(M=make_bin_matrix(feminist_data['M']),
                                   ua='users',
                                   exogenous_ranks_sorted=feminist_data['user_exogenous_ranks'],
                                   user_or_art_dict=feminist_data['user_dict'],
                                   index_function=lambda x: (x-25)/12.5, 
                                   title='Grid Search for User of Feminist Writers',
                                   do_plot=True)


# Note that the white parts of the gridsearch are where the Spearman rho value was not significant using 0.05 as a threshold.

# In[60]:

print('Optimizing points from gridsearch', 
      'rho:', user_spearman['spearman'], 
      'alpha', user_spearman['alpha'], 
      'beta', user_spearman['beta'] )


# Well it looks like this optimising point occurs on the boundary. But we can see also that $\alpha = 0, \beta < 1$ seems to be an optimising solution set ripe for further investigation. 
# 
# #Conclusion
# I hope this allows you to see how Method of Reflections works, and - importantly - how to translate it into different domains. Anywhere you have a bi-partite network you can start ranking nodes using this technique! And if you have some exogenous data, you can also calibrate your model. Please alert me if there are any mistakes in here.
# 
# ##License
# 
# The MIT License (MIT)
# 
# Copyright © 2014 Max Klein aka notconfusing‽
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
