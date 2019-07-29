## A Hierarchical Latent Concept to Paraphrase Variational Autoencoder 

The journals about this project is moved to [this link](https://github.com/Francix/Deep-Generative-Models-for-Natural-Language-Processing/blob/master/README.md), as a reading list 

## Results - Quora

Models                 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Rouge-1 | Rouge-2 | Rouge-L 
------                 | ------ | ------ | ------ | ------ | ------- | ------- | -------
seq2seq                | 51.34  | 36.88  | 28.08  | 22.27  | 52.66   | 29.17   | 50.29
seq2seq-attn           | 53.24  | 38.79  | 29.56  | 23.34  | 54.71   | 30.68   | 52.29
beta-vae, beta = 1e-3  | 43.02  | 28.60  | 20.98  | 16.29  | 41.81   | 21.17   | 40.09
beta-vae, beta = 1e-4  | 47.86  | 33.21  | 24.96  | 19.73  | 47.62   | 25.49   | 45.46
bow-hard               | 33.40  | 21.18  | 14.43  | 10.36  | 36.08   | 16.23   | 33.77 
latent-bow-topk        | 54.93  | 41.19  | 31.98  | 25.57  | 58.05   | 33.95   | 55.74
latent-bow-gumbel      | 54.82  | 40.96  | 31.74  | 25.33  | 57.75   | 33.67   | 55.46
cheating-bow           | 72.96  | 61.78  | 54.40  | 49.47  | 72.15   | 52.61   | 68.53

note: strictly, we should call this cross-aligned VAE 

## Results - MSCOCO

Models                 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Rouge-1 | Rouge-2 | Rouge-L 
------                 | ------ | ------ | ------ | ------ | ------- | ------- | -------
seq2seq                | 69.61  | 47.14  | 31.64  | 21.65  | 40.11   | 14.31   | 36.28
seq2seq-attn           | 71.24  | 49.65  | 34.04  | 23.66  | 41.07   | 15.26   | 37.35
beta-vae, beta = 1e-3  | 68.81  | 45.82  | 30.56  | 20.99  | 39.63   | 13.86   | 35.81
beta-vae, beta = 1e-4  | 70.04  | 47.59  | 32.29  | 22.54  | 40.72   | 14.75   | 36.75
bow-hard               | 48.14  | 28.35  | 16.25  | 9.28   | 31.66   | 8.30    | 27.37
latent-bow-topk        | 72.60  | 51.14  | 35.66  | 25.27  | 42.08   | 16.13   | 38.16
latent-bow-gumbel      | 72.37  | 50.81  | 35.32  | 24.98  | 42.12   | 16.05   | 38.13
cheating-bow           | 80.87  | 65.38  | 51.72  | 41.48  | 45.54   | 20.57   | 40.97

## Results - MSCOCO - Detailed

Models                                      | PPL   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 
------                                      | ---   | ------ | ------ | ------ | ------ 
seq2seq                                     | 4.36  | 69.61  | 47.14  | 31.64  | 21.65
seq2seq-attn                                | 4.88  | 71.24  | 49.65  | 34.04  | 23.66
beta-vae, beta = 1e-3                       | 3.94  | 68.81  | 45.82  | 30.56  | 20.99 
beta-vae, beta = 1e-4                       | 4.12  | 70.04  | 47.59  | 32.29  | 22.54
bow-hard                                    | 19.13 | 48.14  | 28.35  | 16.25  | 9.28
latent-bow-topk                             | 4.75  | 72.60  | 51.14  | 35.66  | 25.27
latent-bow-gumbel                           | 4.69  | 72.37  | 50.81  | 35.32  | 24.98
cheating-bow                                | 15.65 | 80.87  | 65.38  | 51.72  | 41.48
latent-bow-memory-only                      | 
seq2seq-attn top2 sampling                  | 
bow-seq2seq, enc baseline                   | -     | 63.39  | 40.31  | 24.40  | 14.76 
bow-seq2seq, ref baseline                   | -     | 76.09  | 49.90  | 31.79  | 20.41 
bow, predict all para bow                   | -     | 64.44  | 41.26  | 25.90  | 16.47
bow, predict all para bow exclude self bow  |
hierarchical vae                            |

Models                                      | Rouge-1 | Rouge-2 | Rouge-L 
------                                      | ------- | ------- | -------
seq2seq                                     | 40.11   | 14.31   | 36.28
seq2seq-attn                                | 41.07   | 15.26   | 37.35
beta-vae, beta = 1e-3                       | 39.63   | 13.86   | 35.81
beta-vae, beta = 1e-4                       | 40.72   | 14.75   | 36.75
bow-hard                                    | 31.66   | 8.30    | 27.37
latent-bow-topk                             | 42.08   | 16.13   | 38.16
latent-bow-gumbel                           | 42.12   | 16.05   | 38.13
cheating-bow                                | 45.54   | 20.57   | 40.97
seq2seq-attn top2 sampling                  | 
latent-bow-memory-only                      | 

Models                                      | Dist-1 | Dist-2 | Dist-3
------                                      | ------ | ------ | ------
seq2seq                                     | 689    | 3343   | 7400 
seq2seq-attn                                | 943    | 4867   | 11494
beta-vae, beta = 1e-3                       | 737    | 3367   | 6923
beta-vae, beta = 1e-4                       | 1090   | 5284   | 11216
bow-hard                                    | 2100   | 24505  | 71293
latent-bow-topk                             | 1407   | 7496   | 17062
latent-bow-gumbel                           | 1433   | 7563   | 17289
cheating-bow                                | 2399   | 26963  | 70128
seq2seq-attn top2 sampling                  | 
latent-bow-memory-only                      | 

Models                                      | IN-BLEU-1 | IN-BLEU-2 | IN-BLEU-3 | IN-BLEU-4 | Jaccard Dist 
------                                      | --------- | --------- | --------- | --------- | ------------ 
seq2seq                                     | 46.01     | 28.17     | 18.41     | 12.76     | 33.74 
seq2seq-attn                                | 49.28     | 32.23     | 22.19     | 16.06     | 37.60
beta-vae, beta = 1e-3                       | 44.92     | 26.82     | 17.34     | 12.02     | 32.41
beta-vae, beta = 1e-4                       | 46.97     | 29.07     | 19.33     | 13.68     | 34.42
bow-hard                                    | 27.62     | 14.31     | 7.59      | 4.06      | 21.08
latent-bow-topk                             | 51.22     | 34.36     | 24.31     | 18.04     | 39.25
latent-bow-gumbel                           | 
cheating-bow                                | 34.95     | 18.98     | 10.79     | 6.41      | 24.85
seq2seq-attn top2 sampling                  | 
latent-bow-memory-only                      | 
bow-seq2seq, enc baseline                   | 41.40     | 25.31     | 15.78     | 10.13     | -
bow-seq2seq, ref baseline                   | 29.56     | 13.95     | 7.11      | 3.83      | -
bow, predict all para bow                   | 49.07     | 31.17     | 20.55     | 14.18     | -
bow, predict all para bow exclude self bow  |
hierarchical vae                            |


Sentence samples - seq2seq-attn
* I: Five slices of bread are placed on a surface .
* O: A bunch of food that is sitting on a plate .
* I: A wooden floor inside of a kitchen next to a stove top oven . 
* O: A kitchen with a stove , oven , and a refrigerator . 
* I: Four horses pull a carriage carrying people in a parade .
* O: A group of people riding horses down a street .

Random Walk samples - seq2seq-attn
* I: A man sitting on a bench reading a piece of paper 
* -> A man is sitting on a bench in front of a building 
* -> A man is standing in the middle of a park bench 
* -> A man is holding a baby in the park 
* -> A man is holding a baby in a park 
* -> A man is holding a baby in a park 
* I: A water buffalo grazes on tall grass while an egret stands by 
* -> A large bison standing in a grassy field 
* -> A large buffalo standing in a field with a large green grass 
* -> A bison with a green grass covered in green grass 
* -> A large bison grazing in a field with a green grass covered field 
* -> A large bison grazing in a field with a large tree in the background

## Project Vision 

* "Use probabilistic models where we have inductive bias; Use flexible function approximators where we do not."
* This project aims to explore effective Generative Modeling techniques for Natural Langauge Generation

* Two paths
  1. Improving text generation diversity by injecting randomness (or by anything else)
      * Existing text generation models tend to produce repeated and dull expressions from fixed learned modes. 
      * E.g. "I do not know" for any questions in a Question Answering system. 
      * With MLE training, models usually converge to the local maximal which is dominated by the most frequent patterns, thus losing text variety. 
      * We aim to promote text diversity by injecting randomness. 
      * \# NOTE: many existing works do this by using adversarial regularization (Xu et.al., Zhang et.al.) but I want to utilize the randomness of VAE. This idea is not so main-stream so I think I should do some prelimilary verification. 
      * \# NOTE: I have had this idea since last year but have not seem any work about it. So if the prelimilary experiments do not work I may switch back to the existing line. 
  2. Language generation guided by global semantics 
      * Many recent works incorporte global semantic signals (e.g. topics) into sentence generation systems with latent variable models. 
      * These models exhibit many advantages such as better generation quality (but also can be worse honestly), making the generation controllable (which is desirable for decades), and improving interpretability (but sometimes compromises quality). 
      * This work explore the new methods to utilize global semantic signals with latent variable models to improve the downstream generation quality such as language variety. 
  * \# NOTE: These two topics are the most compelling in my mind, but I cannot decide which one is more practical at this time (Feb 06 2019). Will do a survey this week and decide next week. 

* Methods(tentative):
  * Every time one wants to say something, he will have certain _concepts_ in his mind. e.g. "lunch .. burger .. good"
  * At this stage, this _concept_ is not a sentence yet, it is a concept in his mind, he has not say it yet.
  * One has many ways to say this sentence, all the sentences are to some extent different from each other, but they all convey the same meaning. They are _paraphrases_ to each other. 
  * We can think of different sentence realization of this _concept_ as different samples from the same distribution. 
  * Because of stochasticity, each sample is different than each other, which is to say, **stochasticity induces language diversity**
  * Our idea is to use stochasticity to model language diversity.
  * We model one _concept_ as a Gaussian 
  * We model different ways _realization_ of this concept as a mixture Gaussian, each component share the _concept_ Gaussian as their prior. 
  * Given a sentence, we recover the mixture Gaussian, then we use different samples from the mixture to get different paraphrase of that sentence. -- This will require us to reparameterize through Gaussian Mixture, see (Grave 16).  

* Assumptions
  * Simgle Gaussian cannot model stochasticity because of posterior collpse -- TO BE VERIFIED (but I think I have done this before, not 100% sure)

* Goal 
  * Effectiveness: we can actually generate paraphrases
    * surface difference: lower BLEU of different paraphrases
    * semantic similarity: use a classifier to give similarity score 

* Vision
  * upper bound: build new effective models (for one focused application.)
  * upper bound: investigating exising methods and gain a deeper understanding (thus giving a position paper). 
  * lower bound: test existing state of the art models and analyse their pros and cons. 
  * lower bound: continuous trial and error and get to know many ways that do not work. 

* Related Works 
  * Text Generation Models (with particular sentence quality objective)
  * Sentence Variational Autoencoders 
  * Adversarial Regularization for Text Generation 

## Code structures 
* AdaBound.py
* config.py
* controller.py
* data_utils.py
* hierarchical_vae.py
* lm.py
* main.py
* seq2seq.py
* similarity.py 
* vae.py 
