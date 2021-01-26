import numpy as np
import pandas as pd
import os
from flair.embeddings import FastTextEmbeddings, FlairEmbeddings
from flair.embeddings import WordEmbeddings, BytePairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings
from flair.embeddings import DocumentPoolEmbeddings
from flair.data import Sentence
from ipyaggrid import Grid
from ipyaggrid import get_license
from ipyaggrid.magics import CustomMagics




stacked_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward'),
                                        FlairEmbeddings('news-backward'),
                                        WordEmbeddings('en-news') # FastText embeddings over news/wkp
                    
                                       ])

doc_embeddings = DocumentPoolEmbeddings([stacked_embeddings])




def cosine_sim(A, B):
    """computes the cosine similarity between
    two arrays."""
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))









##### Load in the BLS data

dfoesm19nat = pd.read_excel('BLS/oesm19nat/national_M2019_dl.xlsx',
            sheet_name='national_M2019_dl')
dfoesm19natdesc = pd.read_excel('BLS/oesm19nat/national_M2019_dl.xlsx',
        sheet_name='Field Descriptions')
dfoesm19nat['Occupation text'] = dfoesm19nat['occ_title']
dfvdata = pd.read_csv('BLS/dfoesm19nat_vectors.csv')
dfvdata.set_index('index',inplace=True)

##
dmwe = pd.read_csv('BLS/mwe-2019complete.csv')
dfvecs_mwe = pd.read_csv('BLS/mwe_vectors.csv')
dfvecs_mwe.set_index('index',inplace=True)


##
dfgender = pd.read_csv('BLS/dfgender.csv')
dfvecs_gender = pd.read_csv('BLS/gender_vectors.csv')
dfvecs_gender.set_index('index',inplace=True)

##
#dfgeo = pd.read_csv('BLS/dfgeo.csv',
#            converters={'area':str}
#             )

dfbiggeo = pd.read_csv('BLS/dfbiggeo.csv',
             converters={'State':str,
                    'State abbreviation':str,
                    'area':str},
                      low_memory=False)
dfvecs_geo = pd.read_csv('BLS/geo_vectors.csv')
dfvecs_geo.set_index('index',inplace=True)



source_dict = {'national': (dfoesm19nat, dfvdata),
              'model wage estimates': (dmwe, dfvecs_mwe),
              'gender breakdown': (dfgender, dfvecs_gender),
              'MSA': (dfbiggeo, dfvecs_geo)}



def get_cosines(jobtitle, source_title):
    """takes a jobtitle string and the source description
    and returns the cosine similarity scores along with the 
    original source."""
    query = Sentence(jobtitle)
    doc_embeddings.embed(query)
    sims = []
    source_data, source_vecs = source_dict[source_title]
    dfdata = source_data.copy()
    assert 'Occupation text' in dfdata.columns
    dfvecs = source_vecs.copy()
    for i in range(len(dfvecs.index)):
        b = cosine_sim(query.get_embedding().numpy(),
                dfvecs.loc[dfvecs.index[i]].values)
        c = dfvecs.index[i]
        sims.append([jobtitle, b, c])
    dg = pd.DataFrame.from_records(sims)
    dg.columns = ['word','cosine','target']
    dg.sort_values('cosine',ascending=False, inplace=True)
    cosine_dict = dg.set_index('target')['cosine'].to_dict()
    if source_title == 'MSA':
        dfdata['area_detail'] = dfdata['area_title']
    dfdata['target_title'] = dfdata['Occupation text']
    dfdata['cosine'] = dfdata['Occupation text'].map(cosine_dict)
    dfdata['query'] = jobtitle        
    try:
        del dfdata['Sentence']
    except:
        pass
    dfdata.sort_values('cosine',ascending=False, inplace=True)
    
    return dfdata[dfdata.columns[::-1]]


########## ipyagrid functions

def show_grid(ds):
    """takes the dataframe returned from a
    call to get_cosines and presents
    and interactive grid that supports
    sorting, filtering, and export."""
    column_defs = [{'field': c} for c in ds.columns[:]]
    grid_options = {
    'columnDefs' : column_defs,
    'enableColumnResize': True,
    'enableSorting': True,
    'enableFilter': True,
    'enableColResize': True,
    'enableRangeSelection': True,
    'animateRows': True,
    }
    g = Grid(grid_data=ds[ds.columns[:]],
         #columns_fit='size_to_fit',
         columns_fit='auto',
         grid_options=grid_options,
         theme='ag-theme-blue',
         export_csv=True,
         export_excel=True)
    return g


def total_treament(jobtitle, source_title, n=5):
    """same as get_cosines, but returns the
    interacative grid with export controls
    enabled. returns the data for the top n 
    target titles"""
    if source_title not in list(source_dict.keys()):
        print(f"source must be one of {list(source_dict.keys())}")
        return None
    ds = get_cosines(jobtitle, source_title)
    dq = ds[['cosine','target_title']].drop_duplicates()
    good_targets = dq['target_title'].values[:n]
    ds = ds[ds['target_title'].isin(good_targets)].copy()
    ds.sort_values('cosine',ascending=False,inplace=True)
    g = show_grid(ds)
    return g


def total_msa(jobtitle, state_abv, n=5):
    """same as get_cosines, but returns the
    interacative grid with export controls
    enabled. returns the data for the top n 
    target titles"""
    if state_abv not in list(dfbiggeo['State abbreviation'].unique()):
        print(f"source must be one of {list(bls.dfbiggeo['State abbreviation'].unique())}")
        return None
    ds = get_cosines(jobtitle, 'MSA')
    dq = ds[['cosine','target_title']].drop_duplicates()
    good_targets = dq['target_title'].values[:n]
    ds = ds[ds['target_title'].isin(good_targets)].copy()
    ds = ds[ds['State abbreviation'] == state_abv].copy()
    ds.sort_values('cosine',ascending=False,inplace=True)
    
    g = show_grid(ds)
    return g


def all_sources():
    query = input("Enter the jobtitle:")
    message = f"Enter the BLS source type {list(source_dict.keys())}"
    source_data = input(message)
    topn = input("Enter the number of top matching title rows to return")
    return total_treament(query,source_data,n=int(topn))
    
    
def msa_detail():
    query = input("Enter the jobtitle:")
    message = f"Enter the state abbreviation {dfbiggeo['State abbreviation'].unique()}"
    state_abv = input(message)
    topn = input("Enter the number of top matching title rows to return")
    return total_msa(query,state_abv,n=int(topn))


