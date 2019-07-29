[[TOC]]

----

## Latent models for NLP 

#### A Tutorial on Deep Latent Variable Models of Natural Language, EMNLP 18 
* Yoon Kim, Sam Wiseman and Alexander M. Rush

## Paraphrase and Language Diversity 

#### Neural paraphrase generation with stacked residual lstm networks


#### DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text, EMNLP 18 
* Jingjing Xu, Xuancheng Ren, Junyang Lin, Xu Sun

#### Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization, NIPS 18
* Yizhe Zhang, Michel Galley, Jianfeng Gao, Zhe Gan, Xiujun Li, Chris Brockett, Bill Dolan

#### Paraphrase Generation with Deep Reinforcement Learning, EMNLP 18 
* Zichao Li, Xin Jiang, Lifeng Shang, Hang Li

#### A Deep Generative Framework for Paraphrase Generation, AAAI 18
* Ankush Gupta, Arvind Agarwal, Prawaan Singh, Piyush Rai 

## Topic-aware Langauge Generation

#### TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency, ICLR 17 
* Adji B. Dieng, Chong Wang, Jianfeng Gao, John William Paisley

#### Topic Compositional Neural Language Model, AISTATS 18 
* Wenlin Wang, Zhe Gan, Wenqi Wang, Dinghan Shen, Jiaji Huang, Wei Ping, Sanjeev Satheesh, Lawrence Carin

#### Topic Aware Neural Response Generation, AAAI 17 

## Variational Inference, NLP Side 

#### Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder
* Caio Corro, Ivan Titov

#### Spherical Latent Spaces for Stable Variational Autoencoders, EMNLP 18 
* Jiacheng Xu and Greg Durrett 

#### Semi-amortized variational autoencoders, ICML 18 
* Yoon Kim, Sam Wiseman, Andrew C. Miller, David Sontag, Alexander M. Rush
* The **posterior collapse** phenomenon: the variational posterior collapses to the prior and the generative model ignores the latent variable (Dispite all the other stuffs in the intro, I think this is the most important point/ motivation of this paper since the whole NLP community suffer from this for a long time). 
* SVI: view the variational posterior as a model parameter, optimize over is (i.e. the posterior dist. parameter)
* AVI: view the variational posterior as a output of the recognition network (rather than the model parameter), Optimize the recognition network. 
* Semi-armortized VAE: first use a recognition network to predict the variational parameter (the armortized part), then optimize over this parameter (stochastic part.)
* The implementation heavily involves optimization techniques/ tricks. 
* Experiments: higher KL (indicating that latent variables are not collepsed) and lower ppl (performance metrics). 
* Saliency analysis: a visualization of the relationship between the latent variable and the input/ output, as an example of interpretability (or just random guess and coincidence, who knows). 

#### Neural variational inference for text processing, ICML 16 
* Yishu Miao, Lei Yu, Phil Blunsom

#### Lagging Inference Networks and Posterior Collapse in Variational Autoencoders, ICLR 19 
* Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick

#### Avoiding Latent Variable Collapse with Generative Skip Models, AISTATS 19 
* Adji B. Dieng, Yoon Kim, Alexander M. Rush, David M. Blei

#### Improved Variational Autoencoders for Text Modeling using Dilated Convolutions, ICML 17 
* Zichao Yang, Zhiting Hu, Ruslan Salakhutdinov, Taylor Berg-Kirkpatrick

## Variational Inference, ML Side 

#### Stochastic Backpropagation through Mixture Density Distributions, Arxiv 16
* Alex Graves
* This paper gives a method for reparameterize Gaussian Mixture 

#### Auto-Encoding Variational Bayes, Arxiv 13 
* Diederik P. Kingma, Max Welling

#### Variational Inference: A Review for Statisticians, Arxiv 18
* David M. Blei, Alp Kucukelbir, Jon D. McAuliffe 

## Papers discussed in class

#### Reparameterizing the Birkhoff Polytope for Variational Permutation Inference, AISTATS 18 
* Scott W. Linderman, Gonzalo E. Mena, Hal Cooper, Liam Paninski, John P. Cunningham
* The Birkhoff Polytope is the set of doubly-stochastic matrix 
* A doubly-stochastic matrix is a non-negative square matricx whose rows and columns sum to one. 
* A permutation matrix is a matrix where there is only one 1 in each row and column. It is a special type of doubly-stochastic matrices. Doubly-stochastic matrices are continuous versions of permutation matrices, Thus can be used as continuous relaxation of permutation matrix.
* This paper gives a method about continuously transforming a weight matrix to a doubly-stochastic matrix, then anneal this doubly-stochastic matrix to a permutation matrix. All the process differentiable. 
* The goal is to learn (the distribution) of the permutation matrix (for matching, ranking .etc).  
* The second method for learning the distribution of the permutation matrix is using a rounding procedure. 
* This rounding procedure involves: 1. use the Sinkhorn-Knopp algorithm to map a mean (weight) matrix to a doubly-stochastic matrix (but why we should do this?) 2. The sampled matrix = mean matrix + var matrix * noise matrix 3. Round the sampled matrix to the nearest permutation matrix using the Hungarian algorithm. 
* The above two approaches are all differentiable so that can be incorporated into a end to end system
* Experiments show the later is better in practice. 

#### Differentiable Subset Sampling
* Sang Michael Xie and Stefano Ermon
* The gumbel-softmax gives sample of a single entry from a set. 
* Instead of sampling one single entry, we want to sample a subset of size k from a set of size n. How to make this procedure differentiable? 
* Inspired by the weighted reservoir sampling, we construct Gumbel-weights for each entry. (Also recall the relationship of Uniform dist., Exponential dist., Gumbel dist., and Discrete dist. we discussed last week.)
* Then we use a differentiable top-k procedure to get a k-hot vector. This procedure repeat softmax k times, after each step, it set the weight of the previously sampled entry to be -inf (softly). I think this procedure is smart. 

#### Reducing Reparameterization Gradient Variance
* Andrew Miller, Nicholas Foti, Alexander D'Amour, and Ryan Adams 

#### The Generalized Reparameterization Gradient, NIPS 16 
* Francisco J. R. Ruiz, Michalis K. Titsias and David M. Blei

#### Tighter Variational Bounds are Not Necessarily Better, ICML 18 
* Tom Rainforth, Adam R. Kosiorek, Tuan Anh Le, Chris J. Maddison, Maximilian Igl, Frank Wood and Yee Whye Teh 

#### Fixing a Broken ELBO

#### β-VAE: Learning Basic Visual Concepts With A Constrained Variational Framework

#### Disentangling Disentanglement in Variational Auto-Encoders

#### Emergence of Invariance and Disentanglement in Deep Representations 