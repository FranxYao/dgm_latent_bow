
# The Latent Bag of Words Model 

Implementation of Yao Fu, Yansong Feng and John Cunningham, _Paraphrase Generation with Latent Bag of Words_. NeurIPS 2019. [paper](https://github.com/FranxYao/dgm_latent_bow/blob/master/doc/latent_bow_camera_ready.pdf) 

<img src="etc/sample_sentences.png" alt="example"
	title="Example" width="600"  />

As is shown in the above example, given a source sentence, our model first infers the neighbor words of each source words, then sample a bag of words from the neighbors, then generate the paraphrase based on the sampled words 

For more background about deep generative models for natural language processing, see the [DGM4NLP](https://github.com/FranxYao/Deep-Generative-Models-for-Natural-Language-Processing) journal list. 

## Reproduce 

```bash 
mkdir models
mkdir outputs
cd src
python3 main.py 

# quicker start
python3 main.py --batch_size=5 --train_print_interval=10

# Monitor training:
loss: 9.1796  -- total loss
enc_loss: 3.1693  -- BOW NLL loss
dec_loss: 6.0103  -- Decoder loss 
precision_confident: 0.4794  -- BOW precision on confident = most confident word neighbors 
recall_confident: 0.1727  -- BOW recall on confident = most confident word neighbors
precision_topk: 0.1186  -- BOW percision on topk = tok predicted word neighbors
recall_topk: 0.2387  -- BOW recall on topk = tok predicted word neighbors
```

May need to install nltk stopwords first, just follow the prompt 

## Data 

We use the MSCOCO(17) dataset and the Quora dataset. The Quora dataset is provided in the data/ folder. The MSCOCO dataset can be downloaded from its offical website

## Code Structure

The core implementation is in the following files: 

* config.py 
* main.py 
* controller.py 
* latent_bow.py 

## Others

There are certain codes about testing the Wikibio dataset. These part of the code is not included in the paper, its just for tesing the data-to-text task. So the published part might be incomplete. If you do want to extend the model to data-to-text, feel free to contact me. 