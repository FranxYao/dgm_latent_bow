# The Latent Bag of Words Model 

Implementation of Yao Fu, Yansong Feng and John Cunningham, _Paraphrase Generation with Latent Bag of Words_. NeurIPS 2019. [paper](https://github.com/FranxYao/dgm_latent_bow/blob/master/doc/latent_bow_camera_ready.pdf) 

<img src="etc/sample_sentences.png" alt="example"
	title="Example" width="600"  />

As is shown in the above example, given a source sentence, our model first infers the neighbor words of each source words, then sample a bag of words from the neighbors, then generate the paraphrase based on the sampled words 

For more background about deep generative models for natural language processing, see the [DGM4NLP](https://github.com/FranxYao/Deep-Generative-Models-for-Natural-Language-Processing) journal list. 

## Reproduce 

```bash 
cd src
python main.py 
```

## Data 

We use the MSCOCO(17) dataset and the Quora dataset. 

## Code Structure

The core implementation is in the following files: 

* config.py 
* main.py 
* controller.py 
* latent_bow.py 