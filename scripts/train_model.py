#!/usr/bin/env python
# coding: utf-8

# In[254]:


import pandas as pd
import numpy as np
import os
import json
import simplejson
import string
from gensim.models import FastText
import pickle 


# In[257]:


def text_clean( txt ):
    #txt = str(txt)
    return txt 
def get_embedding( text ,  model , N = 200 ):

    text = text_clean( text )

    embedding = np.zeros(  ( N ) )
    words = text.split(" ")
    for word in words:
    
        #if word in model:
        emb =  model.wv[word]
        embedding += emb
        #else:
        #embedding += np.zeros( (N) )

    embedding /= len( words )

    return embedding
def norm_vec(  a ):
    # function to normalize vectors 

    a = a / np.sqrt( (a*a).sum(axis = 1 ) ).reshape( a.shape[0]  , 1 )

    return  np.nan_to_num( a ) 


# In[258]:


def create_corpurs():
    
    # extracts text data from the siemems files
    files = os.listdir("../data/")
    ind = 0
    mapping = {}
    ids2text = {} 
    data = []
    files_json = [ f for f in files if f.endswith(".json")]

    corpus = ""
    for f in files_json:
        print(f)
        fh = open( "../data/" + f )
        #print( type(data))
        #print( data[:1000] )
        json_data = simplejson.load( fh )

        for req in json_data:

            if req["requirement_type"] != "DEF":

                if "text" in req:
                    txt = req["text"]
                if txt is None:
                    continue
                txt = txt.lower()
                txt = txt.strip() 
                txt = txt.replace("\t" , " ")
                txt = txt.encode("utf-8" , "ignore")
                #txt = str(txt)
                txt = txt.translate( string.maketrans("",""), string.punctuation  )


                txt = ' '.join(txt.split())
                txt = txt.decode("utf-8")
                corpus += txt + "\n"
                continue
            if "text" in req:
                txt = req["text"]
                if txt is None:
                    continue
                txt = txt.lower()
                txt = txt.strip() 
                txt = txt.replace("\t" , " ")
                txt = txt.encode("utf-8" , "ignore")
                txt = txt.translate( string.maketrans("",""), string.punctuation  )

                

                if txt != "":
                    data.append( txt )
                    txt = ' '.join( txt.split())
                    txt = txt.decode("utf-8")
                    corpus +=  txt + "\n"
                    mapping[ ind] = req["id"]
                    ids2text[ req["id"] ] = {}
                    ids2text[ req["id"] ]["text"] = txt
                    ids2text[ req["id"] ]["project"] = f
                    ids2text[ req["id"] ]["id"] = req["id"]
                    ind += 1
                    
    return corpus , mapping , data , ids2text
        
    


# In[251]:


corpus , mapping , data , ids2text = create_corpurs()


# In[252]:

print("Training model ... this may take a while")
d = 100
model = FastText(corpus, sg=1, hs=1, size=d, workers=4, iter=5, min_count=10)
print("Model trained")

# In[261]:


vectors = np.zeros(  (  len(data) , d ))

i = 0 
for r in data:
    r = r.decode("utf-8")
    #print(r)
    vec = get_embedding( r , model , d  )

    vectors[ i , : ] = vec
    i += 1
vectors = vectors.astype( np.float32 )

vectors = norm_vec( vectors )


# In[264]:


#save model, mappings , data and ids2text
print( "Dumping outputs")
pickle.dump( file= open("../data/siemens_{}.bin".format( d) , "w") , obj=model , protocol=2 )  # dump model
np.save(arr=vectors ,file="../data/embeddings.npy") # vectors

pickle.dump( file=open("../data/mappings.bin" , "w") , obj=mapping )
pickle.dump( file=open("../data/ids2info.bin" , "w") , obj=ids2text )


