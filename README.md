# Similarity search prototype

A similarity search prototype that is based on word embeddings and fast search in dense vector spaces.


This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

# Technical Description

Using words embeddings and a fast search library, similar requiriments can be searched. 

## The following technologies are used:
- Python
- faiss  https://github.com/facebookresearch/faiss
- gensim https://radimrehurek.com/gensim/
	

## How to Use

Clone the code to your local folder.

There must be one or multiple valid OpenReqJSON files containing the existing requirements in the `/data/` folder for the program to build.

Inside the `/scripts/` folder run:

`python train_model.py` 

that trains and constructs the word embeddings model and saves it in the `data` folder. That is, the model takes as input the text data from the above mentioned requiriments and builds suitable vector representations for the words.  

In order to find similarities for a requirement, create  `reqs.txt` file in the `/scripts/` folder that contains the requirement in plain text form (not JSON!). There can be multiple requirements each on their own line. To detect similar requirements for the requirement, run:

`python similaritySimple.py` 

This will produce an `output.json` file in the same folder. For each queried requiriment the top 10 closest candidates are listed.


## Notes for Developers

None at the moment.

## Sources

None

# How to Contribute
See the OpenReq Contribution Guidelines [here](https://github.com/OpenReqEU/OpenReq/blob/master/CONTRIBUTING.md).

# License

Free use of this software is granted under the terms of the [EPL version 2 (EPL2.0)](https://www.eclipse.org/legal/epl-2.0/)
