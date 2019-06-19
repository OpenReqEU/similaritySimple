# Similarity search prototype

A similarity search prototype that is based on word embeddings and fast search in dense vector spaces.


This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

# Technical Description

Using words embeddings and a fast search library, similar requiriments can be searched. 

## The following technologies are used:
- Python
- faiss  https://github.com/facebookresearch/faiss
- gensim https://radimrehurek.com/gensim/
	

## How to Install

Must have valid  project and requirement OpenReqJSON files in the /data/ folder for the program to build.

`inside the /scripts/ folder run:`


python train_model.py 


to train the word embeddings model. This model takes as input the text data from the requiriments and builds suitable vector representations for the words. The model can also infeer out-of-vocabulary words. 

this script will create a series of files that will be used for future searchs. 
create a *reqs.txt* file in the /scripts/ folder, this file contains the requiriments to be queried. One line text requirement per line

inside de /scripts/ folder run:

python similaritySimple.py 

This will produce an output.json file in the same folder with the output. for each queried requiriment the top 10 closest candidates are selected.


## Notes for Developers

None at the moment.

## Sources

None

# How to Contribute
See the OpenReq Contribution Guidelines [here](https://github.com/OpenReqEU/OpenReq/blob/master/CONTRIBUTING.md).

# License

Free use of this software is granted under the terms of the [EPL version 2 (EPL2.0)](https://www.eclipse.org/legal/epl-2.0/)
