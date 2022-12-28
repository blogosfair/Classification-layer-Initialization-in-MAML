---
layout: distill
title: Strategies for Classification Layer Initialization in Model-Agnostic Meta-Learning
description: [Your blog's abstract - a short description of what your blog is about]
date: 2022-12-01
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# must be the exact same name as your blogpost
bibliography: 2022-12-01-classification-layer-initialization-in-maml.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Quick recap on MAML
  - name: Learning a single initialization vector
  - name: Zero initialization
  - name: MAMLs SCL Intuition
  - name: Initialization using prototypes
  - name: What else is there?
  - name: Conclusion & Discussion
---


## Introduction
In a previous study, [Raghu et al. [2020]](#Raghu) found that in model-agnostic meta-learning (MAML) for few-shot classification, nearly all change observed in the network during the inner loop fine-tuning occurs in the linear classification head. The common interpretation is, that the linear head remaps encoded features to the new tasks classes in this phase. In classical MAML, the final linear layer weights are meta-learned as usual. There are problems to this approach though:

First, it is hard to imagine that a single set of optimal weights can be learned. This becomes evident, when looking at class label permutations: Two different tasks can consist of the same classes, only in different order. Thus, well performing weights for the first task, will hardly be very useful for the second task. This manifests in the fact, that MAMLs performance can vary by up to <strong>15%</strong>, depending on class label assignments during testing [[Ye et al. 2022]](#Ye).

Second, more difficult datasets are being suggested as few-shot learning benchmarks, like Meta-Dataset [[Triantafillou et al. 2020]](#Triantafillou). In those, the number of classes per task varies, which makes it even impossible to learn a single set of weights for the classification layer.

Thus, it seems reasonable to think about how to initialize the final classification layer before fine-tuning on a new task. Random initialization might not be optimal, as e.g. unnecessary noisy is added [[Kao et al. 2022]](#Kao). This blogpost will discuss different last layer initialization-strategies. All the presented approaches clearly outperform original MAML.

## Quick recap on MAML
MAML is a well established algorithm in the field of optimization-based meta-learning. Its goal is to find parameters $ \theta $ for a parametric model $f_{\theta}$, that can be efficiently adapted to perform an unseen task from the same task-distribution, using only a few training examples. The pre-training of $ \theta $ is done using two nested loops (bi-level optimization), with meta-training happening in the outer loop, and task-specific fine-tuning in the inner loop. The task-specific fine-tuning is usually done by a few steps of gradient descend:

$$
\begin{equation}
\theta_{i}' = \theta - \alpha\nabla_{\theta}\mathcal{L_{\mathcal{T_{i}}}}(\theta, \mathcal{D^{tr}})
\end{equation}
$$

where $ \alpha $ describes the inner loop learning rate, function $ \mathcal{L_{\mathcal{T_{i}}}} $ a tasks loss, and $ \mathcal{D^{tr}} $ a tasks training-set, whereas the whole task includes a test-set as well: $ \mathcal{T_{i}} = (\mathcal{D_{i}^{tr}}, \mathcal{D_{i}^{test}}) $.

In the outer loop, the meta parameter $ \theta $ is upgraded by back-propagating through the inner loop, to reduce errors made on <span style="color:blue">the tasks test set</span> using the <span style="color:green">fine-tuned parameters</span>:

$$
\begin{equation}
\theta' = \theta - \eta\nabla_{\theta} \sum_{\mathcal{T_{i}} \sim p(\mathcal{T})}^{} \mathcal{L_{\mathcal{T_{i}}}} (\color{green}  {\theta_{i}'}, \color{blue} {\mathcal{D^{test}}}).
\end{equation}
$$

With $ \eta $ being the meta-learning rate. The differentation through the inner loop includes calculating second-order derivatives, and it mainly distincts MAML from just optimizing for a $ \theta $ that minimizes the average task loss.

Note that in practical scenarios, this second order derivation is computationally expensive, and approximation methods like first-order MAML (FOMAML) [[Finn et al., 2017]](#Finn) or Reptile [[Nichol et al., 2018]](#Nichol) are frequently used. In FOMAML, the outer loop update is simply: $\theta' = \theta - \eta\nabla_{\color{red} {\theta'}} \sum_{\mathcal{T_{i}} \sim p(\mathcal{T})}^{}\mathcal{L_{\mathcal{T_{i}}}}(\theta_{i}', \mathcal{D^{test}})$, which avoids differentiating through the inner loop.

Before proceeding, let's prepare ourselves for the next sections by looking at notations that we can use when discussing MAML in the few-shot classification regime: The models output prediction can be described as $ \hat{y} = f_{\theta}(\mathbf{x}) =  \underset{c\in[N]}{\mathrm{argmax}} \; h_{\mathbf{w}} (g_{\phi}(\mathbf{x}), c)$, where we divide our model $ f_{\theta}(\mathbf{x}) $ (which takes an input $\mathbf{x}$) into a feature extractor $g_{\phi}(\mathbf{x})$ and the classifier $h_\mathbf{w}(\mathbf{r}, c)$, parameterized by classification head weight vectors $\\{\mathbf{w} \\}_{c=1}^N$. $\mathbf{r}$ denotes an inputs represenation, $c$ the index of the class we want the output prediction for.

Finally, $ \theta = \\{\mathbf{w_1}, \mathbf{w_1}, ..., \mathbf{w_N}, \phi\\} $, and we're in harmony with our old notation.

## Learning a single initialization vector
The first two MAML-variants we'll look at, approach the initialization task by initializing the classification head weight-vectors uniformly for all classes. In the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Han-Jia Ye & Wei-Lun Chao (ICLR, 2022) [How to train your MAML to excel in few-shot classification](#Ye),
<p></p>

an approach called <strong>Unicorn MAML</strong> is presented. It is explicitly motivated by the effect that different class-label assignments can have. [Ye & Chao](#Ye) report that during testing, vanilla MAML can perform very differently for <ins>tasks with the same set of classes</ins>, which are just <ins>differently ordered</ins>. Namely, they report that classification accuracy can vary up to 15% on the one-shot setting, and up to 8% in the 5-shot setting. This makes MAMLs performance quite unstable.
<br/><br/>

<p align = "center">
<img src="/public/images/2022-06-23-maml_last_layer/perm_final.png" width="80%" height="80%">
</p>

<p align = "center">
<em>Fig.1 Example of MAML and a class label permutation. We can see the randomness introduced, as $\mathbf{w_1}$ is supposed to interpret the input features as "unicorn" for the first task, and as "bee" in the second. For both tasks, the class outputted as a prediction should be the same, as in human perception, both tasks are identical. This, however, is obviously not the case.</em>
</p>

The solution proposed is fairly simple: Instead of meta-learning $N$ weight vectors for the final layer, only learn a <ins>single vector</ins> $\mathbf{w}$ is meta-learned and used to initialize all $ \\{ \mathbf{w} \\}_{c=1}^N $ before the fine-tuning stage.

This forces the model to make random predictions before the inner loop, as $\hat{y_c}= h_{\mathbf{w}} (g_{\phi} (\mathbf{x}), c)$ will be the same for all $c \in [1,...,N ]$.

After the inner loop, the updated parameters have been computed as usual: $ \theta' = \\{\mathbf{w_1}', \mathbf{w_2}', ..., \mathbf{w_N}', \phi'\\} $. The gradient for updating the single classification head meta weight vector $\mathbf{w}$, is just the aggregation of the gradients w.r.t. all the single $\mathbf{w_c}$:

$$
\begin{equation}
\nabla_{\mathbf{w}} \mathcal{L_{\mathcal{T_i}}} (\mathcal{D^{test}}, \theta_i) = \sum_{c \in [N]} \nabla_{\mathbf{w_c}}
\mathcal{L_{\mathcal{T_i}}} (\theta_i, \mathcal{D^{test}})
\end{equation}
$$

This collapses the models meta-parameters to $ \theta = \\{\mathbf{w}, \phi\\} $.
<br/><br/>

<p align = "center">
<img src="/public/images/2022-06-23-maml_last_layer/unicorn_maml_final.png" width="80%" height="80%">
</p>

<p align = "center">
<em>Fig.2 Overview of Unicorn MAML. We can see that class label permutations don't matter anymore, as before fine-tuning, the probability of predicting each class is the same.</em>
</p>

This tweak to vanilla MAML makes Unicorn MAML permutation invariant, as models fine-tuned on tasks including the same categories - just differently ordered - will now yield the same output predictions. Also, the method could be used with datasets, where the number of classes varies, without any further adaptation: It doesn't matter how many classification head weight vectors are initialized by the single meta classification head weight vector.

Furthermore, the uniform initialization in Unicorn-MAML addresses the problem of memorization overfitting [[Yin et al., 2020]](#Yin). The phenomenon describes a scenario, where a single model can learn all the training tasks only from the test data in the outer loop. This leads to a model that learns to perform the training tasks, but also to a model that doesn't do any fine-tuning, and thus fails do generalize to unseen tasks. Again, the uniform initialization of the classification head for all classes forces the model to adapt during fine-tuning, and thus prevents memorization overfitting.

The approach is reported to perform on par with recent few-shot algorithms.

Let's finally think of how to interpret Unicorn MAML: When meta-learning only a single classification head vector, one could say that not a mapping from features to classes is tried to be learned anymore, but a prioritization of features, which seemed to be more relevant for the classification decision across tasks, than others.

## Zero initialization
The second approach for a uniform initialization is proposed in the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Chia-Hsiang Kao et al. (ICLR, 2022) [MAML is a Noisy Contrastive Learner in Classification](#Kao).
<p></p>

[Kao et al.](#kao) modify original MAML, by setting the whole classification head to zero,
before each inner loop. They refer to this MAML-tweak as the <strong>zeroing trick</strong>.

An overview of MAML with the zeroing trick is displayed below:

<p align = "center">
<img src="/public/images/2022-06-23-maml_last_layer/zeroing_trick.png" width="77%" height="77%">
</p>

<p align = "center">
<em>Fig.3 MAML with the zeroing trick applied.</em>
</p>

Note that $S_n$ and $Q_n$ refer to $\mathcal{D_{i}^{tr}}$ and $\mathcal{D_{i}^{test}}$ in this notation.

Through applying the zero initialization, three of the problems addressed by Unicorn MAML, are solved as well: - MAML with the zeroing trick applied leads to random predictions before fine-tuning. This solves the problem of class
label assignment permutations during testing.
- Through the random predictions before fine-tuning, memorization overfitting is prevented as well.
- The zeroing trick makes MAML applicable for datasets with a varying number of classes per task.

Interestingly, the motivation for applying the zeroing trick, stated by [Kao et al.](#kao), is entirely different. In general, [Kao et al.](#kao) want to unveil in what sense MAML encourages its models to learn general-purpose feature representations. They show, that under some assumptions, there is a supervised contrastive learning (SCL) objective underlying MAML. In SCL, label information is leveraged by pulling embeddings belonging to the same class closer together, while increasing the embedding distances of samples from different classes [[Khosla et al. 2020]](#Khosla).

More specifically, they show that the outer-loop update for the encoder follows a noisy SCL loss under the following assumptions:
1. The encoder weights are frozen in the inner-loop (EFIL assumption)
2. There is only a single inner loop update step.<d-footnote>Note that FOMAML technically follows a noisy SCL loss without this assumption. However, when applying the zeroing trick, this assumtion is needed again for stating that the encoder update is following an SCL loss</d-footnote>

A noisy SCL loss means, that cases can occur where the loss forces the model to maximize similarities between embeddings from samples of different classes. The outer-loop encoder loss in this setting contains an "interference term", which causes the model to pull together embeddings from different tasks, or to pull embeddings into a random direction, with the randomness being introduced by random initialization of the classification head. Those two phenomena are termed <ins>cross-task interference</ins>
and <ins>initialization interference</ins>. Noise and interference in the loss vanish when applying the zeroing trick, and the outer-loop encoder loss turns into a proper SCL loss. Meaning that minimizing this loss forces embeddings of the same class/task together, while pushing embeddings from the same task and different classes apart. A decent increase in performance is observed for MAML with the zeroing trick, compared to vanilla MAML.

Those findings are derived using a general formulation of MAML, with a cross-entropy loss, and the details are available in [the paper](#kao) of course. Also, a slightly simpler example is stated, to give an intuition of MAMLs SCL properties. I will briefly summarize it in the following, to share this intuition with you. However, you might also want to [skip](#initialization-using-prototypes) to the next section.

### MAMLs SCL Intuition
To get an intuition of how MAML relates to SCL, let's look at the following setup: an N-way one-shot classification task, using MAML with Mean Squared Error (MSE) between the 1-hot encoded class label and the models prediction. Furthermore, the EFIL assumption is made, the zeroing trick is applied, only a single inner loop update step is used, and only a single task is sampled per batch.

In this setting, the classification heads inner-loop update for a single datapoint looks like this:

$$
\begin{equation}
\mathbf{w}' = \mathbf{w} - \alpha (-g_{\phi} (\mathbf{x}_{1}^{tr}) \mathbf{t}_{1}^{tr\top})
\end{equation}
$$

$\mathbf{t}_1^{tr}$ refers to the one-hot encoded class label belonging to $\mathbf{x}_1^{tr}$. In words, the features extracted for training example $\mathbf{x}_1^{tr}$ are added to column $\mathbf{w}_c$, with $c$ being the index of 1 in $\mathbf{t}_1^{tr}$. For multiple examples, the features of all training examples labeled with class $c$ are added to the $c^{th}$ column of $\mathbf{w}$.

Now, for calculating the models output in the outer loop, the model computes the dot products of the columns $ \\{\mathbf{w} \\}_{c=1}^N $ and the encoded test examples $$ \begin{equation} g_{\phi}(\mathbf{x}_1^{test}) \end{equation} $$ (and takes a softmax afterwards.) To match the one-hot encoded label as good as possible, the dot product has to be large when $$ \begin{equation} \mathbf{t}_1^{test} \end{equation} $$ = $1$ at index $c$, and small otherwise. We can see that the loss enforces embedding similarity for features from the same classes, while enforcing dissimilarity for embeddings from different classes, which fits the SCL objective.

## Initialization using prototypes
A more sophisticated approach for last layer initialization in MAML is introduced in the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Eleni Triantafillou et al. (ICLR, 2020) [Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples](#Triantafillou).
<p></p>

As one might guess from the name, <strong>Proto-MAML</strong> makes use of Prototypical Networks (PNs) for enhancing MAML. Opposite to the two initialization strategies presented above, Proto-MAML does not uniformly initialize the classification head weights before each inner loop for all classes. Instead, it calculates class-specific initialization vectors, based on the training examples. This solves some of the problems mentioned earlier (see [Conclusion & Discussion](#conclusion--discussion)), but also it adds another type of logic to the classification layer.

Let's revise how PNs work, when used for few-shot learning, for understanding Proto-MAML afterwards:

Class prototypes $$ \begin{equation} \mathbf{c}_c \end{equation} $$ are computed by averaging over each classes train example embeddings, created by a feature extractor $g_{\phi}(\mathbf{x})$. For classifying a test example, a softmax over the distances (e.g. squared euclidean distance) between class prototypes $$ \begin{equation} \mathbf{c}_c \end{equation} $$ and example embeddings $g_{\phi}(\mathbf{x}^{test})$ is used, to generate probabilities for each class.

When using the squared euclidean distance, the models output logits are expressed as:

$$
\begin{equation} - \vert \vert g_{\phi}(\mathbf{x}) - \mathbf{c}_c \vert \vert^2 = −g_{\phi}(\mathbf{\mathbf{x}})^{\top} g_{\phi}(\mathbf{x}) + 2 \mathbf{c}_{c}^{\top} g_{\phi}(\mathbf{x}) − \mathbf{c}_{c}^{\top} \mathbf{c}_{c} = 2 \mathbf{c}_{c}^{\top} g_{\phi}(\mathbf{x}) − \vert \vert \mathbf{c}_{c} \vert \vert^2 + constant.
\end{equation}
$$

(Note that the "test" superscripts on $\mathbf{x}$ are left out for clarity.) $−g_{\phi}(\mathbf{x})^{\top} g_{\phi}(\mathbf{x})$ is disregarded here, as it's the same for all logits, and thus doesn't affect the output probabilities. When inspecting the left-over equation, we can see that it now has the shape of a linear classifier. More specifically, a linear classifier with weight vectors $$ \begin{equation} \mathbf{w}_c = 2 \mathbf{c}_c^{\top} \end{equation} $$ and biases $$ \begin{equation} b_c = \vert \vert \mathbf{c}_{c} \vert \vert^2 \end{equation} $$.

Proceeding to Proto-MAML, [Triantafillou et al.](#triantafollou) adapt vanilla MAML by initializing the classification head using the prototype weights and biases, as just discussed. The initialization happens before the inner loop for each task, and the prototypes are computed by MAMLs own feature extractor. Afterwards, the fine-tuning works as usual. Finally, when updating $\theta$ in the outer loop, the gradients flow also through the initialization of $$ \begin{equation}\mathbf{w}_c \end{equation} $$ and $b_c$, which is easy as they fully depend on $$ \begin{equation} g_{\phi}(\mathbf{x})\end{equation} $$.

Note, that because of computational reasons, Triantafillou refer to Proto-MAML as (FO-)Proto-MAML.

With Proto-MAML, one gets a task-specific, data-dependent initialization in a simple fashion, which seems super nice. For computing the models output logits after classification head initialization, dot products between class prototypes and embedded examples are computed, which again seems very reasonable.

One could argue, that in the one-shot scenario, Proto-MAML doesn't learn that much in the inner loop, beside the initialization itself. This happens, as the dot product between an embedded training example and one class prototype (which equals the embedded training example itself for one class) will be unproportionally high. For a k-shot example, this effect might be less, but still there is always one training example embedding within the prototype to compare. Following this thought, the training samples would rather provide a useful initialization of the final layer, than a lot of parameter adaptation. One has to say that Proto-MAML performed quite well in the authors experiments.

### What else is there?
Before proceeding to [Conclusion & Discussion](#conclusion--discussion), some pointers to methods which didn't perfectly fit the topic, but which are closely related:

The first method worth mentioning is called Latent Embedding Optimization (LEO) [[Rusu et al. 2019]](#Rusu). The authors encode the training data in a low dimensional subspace, from which model parameters $\theta$ can be generated. In the example presented, $\theta$ consists only of $\mathbf{w}$, so for the first inner-loop iteration, this would perfectly fit our initialization topic. The low dimensional code is generated using a feed forward encoder, as well as a matching network. Using the matching network allows LEO to consider relations between the training examples of different classes. Very similar classes for example, might require different decision boundaries than more distinct classes, hence the intuition.

LEO deviates from the initialization scheme, however, as optimization is done in the low dimensional subspace, and not on the models parameters directly. It is stated that optimizing in a lower dimensional subspace helps in low-data regimes.

Another related method is called MetaOptNet [[Lee et al. 2019]](#Lee). In this approach, convex base learners, like support vector machines, are used as the classification head. Those can be optimized till convergence, which solves e.g. the problem of varying performance due to random class label assignments.

## Conclusion & Discussion
To conclude, we've seen that a variety of problems can be tackled by using initialization strategies for MAMLs linear classification head, including:
- Varying performance due to random class label assignments
- Ability of MAML to work on datasets where the number of classes per task varies
- Memorization overfitting
- Cross-task interference
- and Initialization interference.

Furthermore, for all the approaches presented, a decent gain in performance is reported in comparison to vanilla MAML. It seems therefore very reasonable to spend some time thinking about last layer initialization.

Looking at the problems mentioned, and variants discussed in more detail, we can state that all the different variants make MAML <ins>permutation invariant with regard to class label assignments</ins>. Unicorn MAML and the zeroing trick solve it by uniform initialization of $\mathbf{w}$. In Proto-MAML, the initialization happens with regard to the class label assignment, so it's permutation invariant as well.

Also, all variants are compatible with <ins>datasets where the number of classes per task varies</ins>. In Unicorn MAML, an arbitrary number of classification head vectors can be initialized with the single meta-learned classification head weight vector. When zero-initializing the classification head, the number of classes per task does not matter as well. In Proto-MAML, prototypes can be computed for an arbitrary number of classes, so again, the algorithm works on such a dataset without further adaption.

Next, Unicorn MAML and the zeroing trick solve <ins>memorization overfitting</ins>, again by initializing $\mathbf{w}$ uniformly for all classes. Proto-MAML solves memorization overfitting as well, as the task-specific initialization of $\mathbf{w}$ itself can be interpreted as fine-tuning.

<ins>Cross-task interference</ins> and <ins>initialization interference</ins> are solved by the zeroing trick. For the other models, this is harder to say, as the derivations made by [Kao et al.](#Kao) are quite case specific. Intuitively, Proto-MAML should solve cross-task interference, as the classification head is reinitialized after each task. Initialization interference is not solved by either ProtoMAML or Unicorn MAML, as random initialization remains.

Note that in a discussion with a reviewer, [Kao et al.](#kao) state that the main results they show are achieved by models which had the zeroing trick implemented, but which didn't follow the EFIL assumption. They argue that using only the zeroing trick still enhances the supervised contrastiveness. This kind of puts their whole theory into perspective, as without the EFIL assumption, MAML with the zeroing trick is neither an SCL algorithm nor a noisy SCL algorithm. Still, noteable performance gains are reported though.

The question arises, whether the whole theoretical background is needed, or whether the zeroing tricks benefit is mainly the uniform initialization, like in Unicorn MAML. It would be nice to see how the single learned initialization vector in Unicorn MAML turns out to be shaped, and how it compares to the zeroing trick. While the zeroing trick reduces cross-task noise and initialization noise, a single initialization vector can weight some features as more important than others for the final classification decision across tasks.

In contrast to the uniform initialization approaches, we have seen Proto-MAML, where class-specific classification head vectors are computed for initialization, based on the training data.

Finally, [Ye et al.](#Ye) compare the performance between Proto-MAML and Unicorn MAML on MiniImageNet and TieredImageNet. Unicorn MAML performs slightly better here, in the one- and five-shot setting. [Kao et al.](#Kao) don't report any particular numbers for their zeroing trick.