# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pywikibot
from pywikibot import pagegenerators
from pywikibot import userlib
import mwparserfromhell as pfh
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import scipy.stats as ss
import operator
import os
import json
import re
import MySQLdb


class bipartite_data():
    def __init__(self, category_name):
        self.category_name = category_name
        self.enwp = pywikibot.Site('en', 'wikipedia')
        
    def load_arts(self):
        cat = pywikibot.Category(self.enwp, self.category_name)
        self.articles = list(cat.articles())
    
        def earliest_revision(article):
            revisions  = list(article._revisions.itervalues())
            timestamps = map(lambda r: r.timestamp, revisions)
            earliest = min(timestamps)
            return earliest
        
        def load_all_revisions(article):
            before_count = len(article._revisions)
            #print "before ", before_count
            if before_count == 0:
                article.getVersionHistory()
            else:
                self.enwp.loadrevisions(page=article, starttime=earliest_revision(article))
            after_count = len(article._revisions)
            #print "after ", after_count
            if before_count == after_count:
                return
            else:
                load_all_revisions(article)
            
        for article in self.articles:
            load_all_revisions(article)
    
    def make_revision_df(self):
        
        def make_rev_dict(article):
            revdict = {rev.timestamp : {'user' : rev.user, 'article' : article.title()} for rev in article._revisions.itervalues()}
            return revdict
        
        self.all_revisions = pd.DataFrame(columns=['user', 'article'], index=pd.TimeSeries())
        
        for article in self.articles:
            self.all_revisions = self.all_revisions.append(pd.DataFrame.from_dict(data=make_rev_dict(article), orient='index'))
            
    def remove_bots(self):
        self.no_bots = self.all_revisions[self.all_revisions['user'].map(lambda username: not re.findall('bot|stat', username, flags=re.IGNORECASE))]

    def remove_min_editors(self, remove_minimum=5):
            
        user_ar = self.no_bots.groupby(by='user')
        edit_sizes = user_ar.size()
        min_editors = edit_sizes[ edit_sizes > remove_minimum]
    
        criterion = self.no_bots['user'].map(lambda user: user in min_editors.index)
        min_revisions = self.no_bots[criterion]
    
        self.sorted_min_revisions = min_revisions.sort(axis=0)
    
    def make_user_art_dicts(self):
        u_grouped = self.sorted_min_revisions.groupby('user')
        a_grouped = self.sorted_min_revisions.groupby('article')
        
        '''these dicts are so we can back translate the indexes in the matrix to real users and articles'''
        users = list(u_grouped.groups.iterkeys())
        articles = list(a_grouped.groups.iterkeys())
        self.user_dict = {username: users.index(username) for username in users}
        self.article_dict = {articlename: articles.index(articlename) for articlename in articles}
        
    def make_contributor_matrix(self):
    
        ua_grouped = self.sorted_min_revisions.groupby(by = ['user', 'article'])
        self.contributor_matrix = np.zeros(shape=(len(self.user_dict), len(self.article_dict)))

        for user_article_tuple, timestamps in ua_grouped.groups.iteritems():
            user_string = user_article_tuple[0]
            article_string = user_article_tuple[1]
            user_index = self.user_dict[user_string]
            article_index = self.article_dict[article_string]
            
            self.contributor_matrix[user_index][article_index] = len(timestamps)
    
    def rank_exogenous_dict(self, exogenous_dict):
        exogenous_dict_sorted = sorted(exogenous_dict.iteritems(), key=operator.itemgetter(1))
        exogenous_ranks_sorted = [(identup[0], exogenous_dict_sorted.index(identup)) for identup in exogenous_dict_sorted]
        return exogenous_ranks_sorted
    
    def make_exogenous_article_metrics(self):            
            exogenous_articles = dict()
            
            for article in self.article_dict.iterkeys():
                exogenous_articles[article] = calculate_article_metric(article, 'articlelength')
            
            self.exogenous_articles_ranked = self.rank_exogenous_dict(exogenous_articles)
    
    def make_exogenous_user_metrics(self):
        exogenous_users = dict()
        conn = MySQLdb.connect(host='enwiki.labsdb', db="enwiki_p", port=3306, read_default_file="~/replica.my.cnf") 
        cursor = conn.cursor() 

        for user in self.user_dict.iterkeys():
            exogenous_users[user] = calculate_edit_hours(user,cursor)

    
        self.exogenous_users_ranked = self.rank_exogenous_dict(exogenous_users)
    
    def save_everything(self):
        directory = self.category_name
        today = str(datetime.date.today())
        double_directory = 'savedata/' + directory + '/' + today + '/'
        
        if not os.path.exists(double_directory):
            os.makedirs(double_directory)
        
        mfilename = double_directory  + 'M' + '.npy'
        mf = open(mfilename, 'w')
        np.save(mf, self.contributor_matrix)
        
        for filename, filedict in {'user_dict': self.user_dict, 
                                   'article_dict': self.article_dict,
                                   'user_exogenous_ranks': self.exogenous_users_ranked, 
                                   'article_exogenous_ranks': self.exogenous_articles_ranked}.iteritems():
        
            path = double_directory  + filename + '.json'
            f = open(path, 'w')
            json.dump(filedict, f)
    
    def do_everything(self):
        self.load_arts()
        self.make_revision_df()
        self.remove_bots()
        self.remove_min_editors()
        self.make_user_art_dicts()
        self.make_contributor_matrix()
        self.make_exogenous_article_metrics()
        self.make_exogenous_user_metrics()
        self.save_everything()

# <codecell>


# <codecell>

#Infonoise metric of Stvilia (2005) in concept, although the implementation may differ since we are not stopping and stemming words, because of the multiple languages we need to handle

def readable_text_length(wikicode):
    #could also use wikicode.filter_text()
    return float(len(wikicode.strip_code()))

def infonoise(wikicode):
    wikicode.strip_code()
    ratio = readable_text_length(wikicode) / float(len(wikicode))
    return ratio

#Helper function to mine for section headings, of course if there is a lead it doesn't quite make sense.

def section_headings(wikicode):
    sections = wikicode.get_sections()
    sec_headings = map( lambda s: filter( lambda l: l != '=', s), map(lambda a: a.split(sep='\n', maxsplit=1)[0], sections))
    return sec_headings

#i don't know why mwparserfromhell's .fitler_tags() isn't working at the moment. going to hack it for now
import re
def num_refs(wikicode):
    text = str(wikicode)
    reftags = re.findall('<(\ )*?ref', text)
    return len(reftags)

def article_refs(wikicode):
    sections = wikicode.get_sections()
    return float(reduce( lambda a,b: a+b ,map(num_refs, sections)))

#Predicate for links and files in English French and Swahili

def link_a_file(linkstr):
    fnames = [u'File:', u'Fichier:', u'Image:', u'Picha:']
    bracknames = map(lambda a: '[[' + a, fnames)
    return any(map(lambda b: linkstr.startswith(b), bracknames))

def link_a_cat(linkstr):
    cnames =[u'Category:', u'Catégorie:', u'Jamii:']
    bracknames = map(lambda a: '[[' + a, cnames)
    return any(map(lambda b: linkstr.startswith(b), bracknames))

def num_reg_links(wikicode):
    reg_links = filter(lambda a: not link_a_file(a) and not link_a_cat(a), wikicode.filter_wikilinks())
    return float(len(reg_links))

def num_file_links(wikicode):
    file_links = filter(lambda a: link_a_file(a), wikicode.filter_wikilinks())
    return float(len(file_links))

def report_actionable_metrics(wikicode, completeness_weight=0.8, infonoise_weight=0.6, images_weight=0.3):
    completeness = completeness_weight * num_reg_links(wikicode)
    informativeness = (infonoise_weight * infonoise(wikicode) ) + (images_weight * num_file_links(wikicode) )
    numheadings = len(section_headings(wikicode))
    articlelength = readable_text_length(wikicode)
    referencerate = article_refs(wikicode) / readable_text_length(wikicode)

    return {'completeness': completeness, 'informativeness': informativeness, 'numheadings': numheadings, 
            'articlelength': articlelength, 'referencerate': referencerate}

def calculate_article_metric(article_name, metric):
    page = pywikibot.Page(enwp, article_name)
    page_text = page.get()
    wikicode = pfh.parse(page_text)
    metrics = report_actionable_metrics(wikicode)
    return metrics[metric]


def calculate_edit_hours(user, cursor):
    starttime = datetime.datetime.now()
    qstring = u'''SELECT rev_timestamp FROM enwiki_p.revision_userindex WHERE rev_user_text like "'''+ user + u'''";'''
    uqstring = qstring.encode('utf-8')
    cursor.execute(uqstring)
    results = cursor.fetchall()
    clean_results = map(lambda t: t[0], results)
    timestamps = map(pywikibot.Timestamp.fromtimestampformat, clean_results)
                                                                                                                                  
    edit_sessions = []
    curr_edit_session = []

    prev_timestamp = datetime.datetime(year=2001, month=1, day=1)


    for contrib in timestamps:
        curr_timestamp = contrib
                                                                                                                                         
 
        if curr_timestamp-prev_timestamp < datetime.timedelta(hours=1):
            curr_edit_session.append(curr_timestamp)
            prev_timestamp = curr_timestamp

        else:
            if curr_edit_session:
                edit_sessions.append(curr_edit_session)
            curr_edit_session = [curr_timestamp]
            prev_timestamp = curr_timestamp

    #finally have to add the curr_edit_session to list                                                                                                         
    if curr_edit_session:
        edit_sessions.append(curr_edit_session)


                                                                                                                                  
    def session_length(edit_session):
        avg_time = datetime.timedelta(minutes=4, seconds=30)
        last = edit_session[-1]
        first = edit_session[0]
        span = last - first
        total = span + avg_time
        return total

    session_lengths = map(session_length, edit_sessions)
    second_lens = map(lambda td: td.total_seconds(), session_lengths)
    total_time = sum(second_lens)

    took = datetime.datetime.now() - starttime
    tooksecs = took.total_seconds()
    print 'timestamps per second: ', len(timestamps)/float(tooksecs)
    #returning total hours                                                                                                                                     
    return total_time / float(3600)

if __name__ == '__main__':
	cats = ['Category:Feminist_writers', 'Category:Works_based_on_Sherlock_Holmes', 'Category:Non-Euclidean_geometry', 'Category:Military_history_of_the_United_States', 'Category:Yoga']
	for cat in cats:
		stime = datetime.datetime.now()
		print 'doing', cat
		bp = bipartite_data(cat)
		bp.do_everything()
		print datetime.datetime.now() - stime, 'time to do ', cat



