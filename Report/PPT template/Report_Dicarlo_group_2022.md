---
marp: true
theme: gaia
paginate: true
# footer: 'Yaoyu Zhang 2022-09-12'
style: |
section a { font-size: 34px;}
math: mathjax
# math: katex
---
<!-- 
_class: lead gaia
_paginate: false -->
![w:3.5cm contrast](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/output.png) 
## Adversarially trained neural representations may already be as robust as corresponding biological neural representations 
#### Zhiwei Bai
#### Shanghai Jiao Tong University
#### 2022-12-04

---
<!-- backgroundColor: white -->
<style scoped>
section ul li {font-size: 32px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## About the author?
![w:22cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101162858.png)
James J.DiCarlo
- Born in 1968, `neuroscientist `
- Peter de Florez Professor, Brain and Cognitive Sciences
- Director, MIT Quest for Intelligence
- **"Aim to understand how a complex network of brain regions underlies our ability to recognize vast numbers of objects and faces rapidly."**
![bg right:30% w:9cm contrast:120%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101162555.png) 

---
<style scoped>
section ol li {font-size: 49px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## Outline
1. **Motivation**: why do we care about adversarial robustness?
2. **Result**: who is more robust between AT-DNNs and primate visual perception?
3. **Method**: how to measure adversarial sensitivity of IT neural sites?



---
$\quad$
$\quad$
$\quad$
# 1. Motivation: why do we care about adversarial robustness?


---
<style scoped>
section ul li {font-size: 36px;}
</style>
<style scoped>
section p {font-size: 25px;}
</style>
## Motivation: DNN is VERY brittle
- Pre-trained ResNet50

![w:22cm contrast:120%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101171831.png)
High confidence (0.996) $\hspace{3.5cm}$ High confidence (0.999)  
![bg right:25% w:6cm contrast:120%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101171950.png) 
#### **Some specific perturbations can easily fool current deep neural networks.**

---
<style scoped>
section ul li {font-size: 36px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## How to create adversarial examples?
- Given data $S = \{(\boldsymbol{x}_j, \boldsymbol{y}_j)\}_{j=1}^{n}$, usual training goal:
$$
\underset{\boldsymbol{\theta}}{\operatorname{min}} R_S(\boldsymbol{\theta}) = \mathbb{E}_S\ell(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}), \boldsymbol{y}):=\frac{1}{n} \sum_{j=1}^n \ell\left(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_j), \boldsymbol{y}_j\right)
$$
- Create adversarial examples: for a fixed $(\boldsymbol{x}, \boldsymbol{y})\in S$,
$$
\underset{\|\boldsymbol{\delta}\| < \epsilon}{\operatorname{max}} \ell\left(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x} + \boldsymbol{\delta}), \boldsymbol{y}\right)
$$
- Targeted attack: $\underset{\|\boldsymbol{\delta}\| < \epsilon}{\operatorname{max}} \left(\ell\left(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x} + \boldsymbol{\delta}), \boldsymbol{y}\right) -\ell\left(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x} + \boldsymbol{\delta}), \boldsymbol{y}_{\mathrm{target}}\right)\right)$

---
<style scoped>
section ul li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## What is adversarial training?
- Usual training goal:
$$
\underset{\boldsymbol{\theta}}{\operatorname{min}} R_S(\boldsymbol{\theta}) = \mathbb{E}_S\ell(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}), \boldsymbol{y}):=\frac{1}{n} \sum_{j=1}^n \ell\left(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_j), \boldsymbol{y}_j\right)
$$
- Adversarial training goal:
$$
\underset{\boldsymbol{\theta}}{\operatorname{min}} \hat{R}_S(\boldsymbol{\theta}) = \mathbb{E}_S\max_{\|\boldsymbol{\delta}\|<\epsilon}\ell(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}+\boldsymbol{\delta}), \boldsymbol{y}):=\frac{1}{n} \sum_{j=1}^n \max_{\|\boldsymbol{\delta}_j\|<\epsilon}\ell\left(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_j + \boldsymbol{\delta}_j), \boldsymbol{y}_j\right)
$$
- $R_S(\boldsymbol{\theta}) \leq \hat{R}_S(\boldsymbol{\theta})$, intuitively $\hat{R}_S(\boldsymbol{\theta})$ is the `worst-case`.

---
<style scoped>
section ul li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## Is adversarial training enough?
- Gold standard of robust perception: visual systems of primates

$\hspace{1.8cm}$![w:21cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/2.svg)

<!-- <center>

```plantuml
@startuml
' left to right direction
top to bottom direction
[Robustness of the best of artificial NNs] as A
[Robustness of visual systems of primates] as B
A <|--|> B

@enduml
```
</center> -->

---
<style scoped>
section ul li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## Do visual systems of primates really robust?
![h:9cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/IMG_0137.JPG) ![h:9cm contrast](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101200210.png)

---
<style scoped>
section ul li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 36px;}
</style>
## Do visual systems of primates really robust?
![h:7.5cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/IMG_0137.JPG) ![h:7.5cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/IMG_0135.JPG)


---
<style scoped>
section ol li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
## Are these adversarial examples in primates widespread or only for a few specific tasks?
In particular, do adversarial examples exist for cat and dog classification tasks?
#### To answer the question, we need answer:
1. How to set $\epsilon$ in $\|\boldsymbol{\delta}\| < \epsilon$ ?
2. "Worst-case" relies on detailed knowledge. How to create adversarial examples?
$$\underset{\|\boldsymbol{\delta}\| < \epsilon}{\operatorname{max}} \ell\left(\boldsymbol{r}(\boldsymbol{x} + \boldsymbol{\delta}), \boldsymbol{y}\right)$$
<!-- ![w:30cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101193554.png) -->

---
<style scoped>
section ol li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
## How to set $\epsilon$ in $\|\boldsymbol{\delta}\| < \epsilon$ ?
- Consider $l_2$ norm $\|\delta\|_2 < \epsilon$

![w:30cm contrast](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221101212823.png)

- Restrict to a specific regime, such as $[1, 10]$

---
<style scoped>
section ol li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
## "Worst-case" relies on detailed knowledge. How to create adversarial examples?
$$\ell(\boldsymbol{\delta}^*):=\underset{\|\boldsymbol{\delta}\| < \epsilon}{\operatorname{max}} \ell\left(\boldsymbol{r}(\boldsymbol{x} + \boldsymbol{\delta}), \boldsymbol{y}\right)$$
1. Black-box attack: without using detailed knowledge.
   - Rely on `random sampling` image perturbation directions, "unlikely to yield good estimates of adversarial sensitivity".

2. Build a  “white-box” model to `estimate` the adversarial example.
   - This paper develops an experimental method to do this.
   - `Lower bound` is enough for this paper's claim.

---
<style scoped>
section ul li {font-size: 33px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
## Which area of neurons to choose?
- V1 (Primary Visual cortex)
  - Support: Primates is more robust
- IT (Inferior Temporal cortex)
  - Support: Artificial NN is more robust
  - Compare with the penultimate layer of DNN
![bg right:51% w:16cm contrast:150%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102110438.png)



---
<style scoped>
section ol li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
## How to quantify robustness?
- Create adversarial examples: for a fixed $(\boldsymbol{x}, \boldsymbol{y})\in S$,
$$\ell(\boldsymbol{\delta}^*):=\underset{\|\boldsymbol{\delta}\| < \epsilon}{\operatorname{max}} \ell\left(\boldsymbol{r}(\boldsymbol{x} + \boldsymbol{\delta}), \boldsymbol{y}\right)$$
- Quantify sensitivity: for a fixed $\boldsymbol{x}\in S_{\boldsymbol{x}}$ and a fixed $i^{\mathrm{th}}$ neural site,
$$
s_i(\boldsymbol{x}, \epsilon) := \underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}}| r_i(\boldsymbol{x} + \boldsymbol{\delta}) - r_i(\boldsymbol{x})|
$$
### $\hspace{1.5cm}$ **Why not $\underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}}\| \boldsymbol{r}(\boldsymbol{x} + \boldsymbol{\delta}) - \boldsymbol{r}(\boldsymbol{x})\|_1$ ?**

---
<style scoped>
section ol li {font-size: 35px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
## Quantify robustness: individual unit level
- Quantify sensitivity: for a fixed $\boldsymbol{x}\in S_{\boldsymbol{x}}$ and a fixed $i^{\mathrm{th}}$ neural site,
$$
s_i(\boldsymbol{x}, \epsilon) := \underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}}| r_i(\boldsymbol{x} + \boldsymbol{\delta}) - r_i(\boldsymbol{x})|
$$
- Marginalizing the image distribution $S_{\boldsymbol{x}}$:
$$
s_i(\epsilon) := \mathbb{E}_{\boldsymbol{x}\sim S_{\boldsymbol{x}}}[s_i(\boldsymbol{x}, \epsilon)]
$$
- Normalized adversarial sensitivity:
$$
\tilde{s}_i(\epsilon) = \dfrac{s_i(\epsilon)}{\sigma_i}, \text{where } \sigma_i = \left(\mathrm{Var}_{\boldsymbol{x}\sim S_{\boldsymbol{x}}}r_i(\boldsymbol{x})\right)^{\frac{1}{2}} 
$$

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section h1 {font-size: 59px;}
</style>
$\quad$
$\quad$
$\quad$
# 2. Result: who is more robust between AT-DNNs and primate visual perception?

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
### Who is more robust?
- The number of IT sites $m=21$
$
\tilde{s}(\epsilon) = \dfrac{1}{m}\sum\limits_{i=1}^{m}\tilde{s}_i(\epsilon)$
- Adversarially trained NN $(l_2\epsilon=3)$, 10-fold smaller
- Grey dashed: standard deviation
- Blue dashed: random pairs of images $\dfrac{1}{C_n^2}\sum\limits_{j\neq k}|r_i(\boldsymbol{x}_j)- r_i(\boldsymbol{x}_k)|/\sigma_i$
![bg right:50% w:15cm contrast:105%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102113024.png)

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
##### What do the adversarial examples of primate IT neurons look like?
$\hspace{1.4cm}$![w:23cm contrast:100%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102193055.png)

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
#### How stable is “category preference”  of each IT neural site?
- Fix a site, identify the most and least preferred categories
- #### **Q: Is "category preference" well-defined?**
- Perform targeted adversarial perturbation
- $\epsilon =2.5$ highly-preferred
- $\epsilon =10$ "super-stimuli"
- Red dashed: preferred images
![bg right:50% w:15cm contrast:105%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102195758.png) 

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 35px;}
</style>
#### Are all IT neurons susceptible, or could the average results be due to just a few strongly modulated neurons?
- Adversarial images can be found on `all recorded IT sites `
- Adversarial images can be found very close to `any clean images`
$\hspace{2cm}$![w:18cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102204549.png)
- Adversarial samples for biological neurons are `dense` in the image space similar to that of artificial neural networks

---
<style scoped>
section ul li {font-size: 32px;}
</style>
<style scoped>
section h1 {font-size: 60px;}
</style>
$\quad$
$\quad$
$\quad$

# 3. Method: how to measure adversarial sensitivity of IT neural sites?


---
<style scoped>
section ul li {font-size: 40px;}
</style>
<style scoped>
section p {font-size: 40px;}
</style>
## Measure adversarial sensitivity of IT neural sites
- Recall goal:
$$
s_i(\boldsymbol{x}, \epsilon) := \underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}}| r_i(\boldsymbol{x} + \boldsymbol{\delta}) - r_i(\boldsymbol{x})|
$$
- Method: Build a  “white-box” model to `estimate` the adversarial example
  - Iteratively generate better lower bound of adversarial example 

<!-- ## Screen a baseline model, criteria:
1. Global representational similarity to IT as measured by CKA
2. Cross-validated linear predictivity for IT responses
3. How well does perturbations targeted toward a model layer transfers to IT neurons without any explicit mapping between the two systems -->

---
<style scoped>
section ol li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
### Screen a baseline model, criteria
1. Global representational similarity to IT as measured by CKA
2. Cross-validated linear predictivity for IT responses
3. How well does perturbations targeted toward a model layer transfers to IT neurons without any explicit mapping between the two systems
$\hspace{2cm}$![w:19cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102215154.png)


---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
## ResNet50
$\quad$
$\quad$
$$
\begin{aligned}
    &1 \text{ (conv)}\\
    +& 3\times 4  \text{ (conv\_block)}\\
    +& 3\times (2+3+5+2) \text{ (identify\_block)}\\
    + & 1\text{ (Fully-connected)}\\
    = & 50 
\end{aligned}
$$
$\hspace{2.5cm}$![bg right:55% w:18cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102214105.png)

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
## Create adversarial examples iteratively
![w:30cm](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102221709.png)

---
<style scoped>
section ul li {font-size: 30px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
#### Details of measure effect of attacks on IT neural sites
- Show both clean and attack images to a `fixating monkey`
- The visual stimuli are presented `8 degrees `over the visual field for 100ms followed by a 100ms `grey mask` as in a standard rapid serial visual presentation (RSVP) task.
- The average temporal separation between a clean image and its perturbed pair is `25 minutes`
- Total of `6 days`. 
- For Figure 1A, we report IT sensitivity from the last day of experiment which sampled `882 unique images` per perturbation $\epsilon$ (i.e. 42 images per neural site).
- Measure the total number spikes between `70ms-170ms` after image presentation. 

---
<style scoped>
section ol li {font-size: 33px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
## Does the method work?
1. Neural perturbation magnitude has an `consistent improvement` over days.
2. The perturbations achieved with our method is `significantly larger` than that achieved with a model-free method.
3. This explains why the field has systematically `underestimated` the sensitivity.
![bg right:42% w:13cm contrast:100%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221102225017.png) 

<!-- ---
<style scoped>
section ul li {font-size: 33px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
## Some specific details: bias
- In the presence of measurement noise, taking the average of the absolute value of response change will result in a `positive bias`.
- For $x_j$ where $j=0,1, \ldots, n$, out of $n$ number of images: 
$$
\begin{aligned}
    E_j\left[\left|r_i\left(x_j\right)-r_i\left(x_j+\delta_{i, j}\right)\right|\right]& =\frac{1}{n} \sum_j \operatorname{sign}\left(r_i\left(x_j\right)-r_i\left(x_j+\right.\right.\left.\left.\delta_{i, j}\right)\right)\left(r_i\left(x_j\right)-r_i\left(x_j+\delta_{i, j}\right)\right.\\
    & \geq \frac{1}{n}\sum_j \text{sign}(f_i(x_j, \theta)-f_i(x_j+\delta_{i,j},\theta))(r_i(x_j)-r_i(x_j+\delta_{i,j})
\end{aligned}
$$
- This estimator becomes `unbiased` if the model of IT site $f_i(x, \theta)$ from the previous day predicted all the directions of neural movement correctly. -->

---
<style scoped>
section ul li {font-size: 33px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
### Introduce multiple methods to drastically improve convergence beyond the basic PGD
1. $100$ `independent runs` for solving the adversarial images
2. Optimizing $\underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}} f_i(\boldsymbol{x} + \boldsymbol{\delta}) - f_i(\boldsymbol{x})$ and $\underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}} f_i(\boldsymbol{x}) - f_i(\boldsymbol{x} + \boldsymbol{\delta})$ `separately` — reduce the chances to be stuck at saddle point. **Why?**
3. `Larger` $\epsilon$ converges faster. First with a ball of radius $2\epsilon$ and finally with one of radius $\epsilon$. **Why?**
4. `Simulated annealing` with restarts: we begin with steps of size $\epsilon$ and reduce them by 10% every time no progress is made.

---
<style scoped>
section ul li {font-size: 33px;}
</style>
<style scoped>
section p {font-size: 30px;}
</style>
## PGD
- Goal:
    $$
    s_i(\boldsymbol{x}, \epsilon) := \underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{max}}| f_i(\boldsymbol{x} + \boldsymbol{\delta}) - f_i(\boldsymbol{x})|
    $$
- Projected Gradient Descent
$$ 
\underset{\|\boldsymbol{\delta}\|_2 < \epsilon}{\operatorname{min}} g(\boldsymbol{\delta}):= -| f_i(\boldsymbol{x} + \boldsymbol{\delta}) - f_i(\boldsymbol{x})|
$$
$$
\begin{gathered}
\tilde{\boldsymbol{\delta}}_{t+1}=\boldsymbol{\delta}_t-\eta \nabla g\left(\boldsymbol{\delta}_t\right) \\
\boldsymbol{\delta}_{t+1}=\underset{{\|\boldsymbol{\delta}\|_2 < \epsilon}}{\mathrm{argmin}}\left\|\tilde{\boldsymbol{\delta}}_{t+1}-\boldsymbol{\delta}\right\|_2
\end{gathered}
$$
![bg right:50% w:15cm contrast:120%](https://cdn.jsdelivr.net/gh/baizhiwei299/images_for_vscode/images/20221103093630.png) 

---
<style scoped>
section ul li {font-size: 33px;}
</style>
<style scoped>
section p {font-size: 50px;}
</style>
## Conclusion
The representations learned by adversarially trained artificial neural networks have already `exceeded` that of the corresponding biological neural representation in terms of their `individual unit level` adversarial robustness.

---
<style scoped>
section ol li {font-size: 40px;}
</style>
<style scoped>
section p {font-size: 38px;}
</style>
### **Paradox**: how is it that primate visual perception seems so robust yet its fundamental units of computation are far more sensitive than expected?
1. Visual object recognition behavior in primate is actually `NOT adversarial robust`
2. There is an unknown `error-correction `mechanism at the `population level` in IT or in a down-stream area that decodes object identity
  
---
<style scoped>
section ul li {font-size: 50px;}
</style>
<style scoped>
section p {font-size: 58px;}
</style>
## Future work & new start
- Population level robustness
- Provides us with a set of standardized procedure

---
$\quad$
$\quad$
# <!-- fit -->Thank you!