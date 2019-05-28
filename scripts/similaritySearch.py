import pandas as pd
import numpy as np
import os
import json
import simplejson
import string
from gensim.models import FastText
import gensim 
import pickle 
import faiss

def clean_text( txt ):

	txt = txt.lower()
	txt = txt.strip() 
	txt = txt.replace("\t" , " ")
	txt = txt.encode("utf-8" , "ignore")
	#txt = str(txt)
	txt = txt.translate( string.maketrans("",""), string.punctuation  )
	return txt


def norm_vec(  a ):
    # function to normalize vectors 

    a = a / np.sqrt( (a*a).sum(axis = 1 ) ).reshape( a.shape[0]  , 1 )

    return  np.nan_to_num( a ) 

def get_embedding( text , model , N = 100 ):

    text = clean_text( text )

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

def load_reqs_file( model ):

	try:
		f = open( "./reqs.txt")
	except:

		print(" File reqs.txt must exists in current directory" )
		return None 

	data = []
	lines = []
	for  l in f:
		emb = get_embedding( l , model )
		lines.append( l )
		data.append( emb  )

	data = np.array( data ).astype( np.float32 )

	# returnd the embeddings for the 
	return data , lines 

def load_model():

	data_full = np.load( "../data/embeddings.npy")
	mappings = pickle.load( open("../data/mappings.bin" , "r") )
	inverse_mappings = {v: k for k, v in mappings.items() }

	id2info = pickle.load( open( "../data/ids2info.bin" , "r" )  )

	#model = gensim.models.fasttext.load_facebook_model( "../data/siemens_200.bin" )
	model = pickle.load( open("../data/siemens_100.bin" , "r"))
	return model , data_full , mappings, inverse_mappings , id2info 

def main():

	model , data , mappings , inverse_mappings , id2info = load_model()
	if data is None:
		print("Requiriments file  not found.")
		return None 
	D = model.wv["sample"].shape[0]
	index = faiss.IndexFlatIP( D )

	#build the index " "
	index.train(  data.astype( np.float32 ) )
	index.add(  data.astype( np.float32 )  )

	queries , lines  = load_reqs_file( model )

	queries = norm_vec( queries )


	final_result = []


	for q , l  in zip( queries , lines  ):

		result = {}
		distances , I = index.search( q.reshape( (1 , -1 )  ) , 10    )
		similar = [ mappings[i] for i in I[0][1:] ]

		result["req_text"] = l 
		result["similar_req"] = []


		for s in similar:

			result["similar_req"].append( id2info[s] )

		final_result.append( result )
		#print( similar)

	with open("./output.json" , "w") as f:

		json.dump( final_result , f  , indent = 2 )
		#f.write(js )

	#print( final_result )
	print("Output created")
	return True


if __name__ == "__main__":

	main()