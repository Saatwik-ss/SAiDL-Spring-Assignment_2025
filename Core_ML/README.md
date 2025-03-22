# Robust Learning with Noisy CIFAR-10 Labels Using Normalized Losses and the APL Framework

## Introduction
In real-world machine learning tasks, datasets often contain **noisy labels**, which can degrade model performance by leading to overfitting. To address this issue, robust loss functions have been proposed to enhance **noise tolerance** while maintaining model performance. However, a key challenge is the trade-off between **robustness** and **accuracy on clean data**. 

This project explores normalized loss functions** and the Active-Passive Loss (APL) framework to improve model robustness under label noise. The experiments include training models on CIFAR-10 with both symmetric and asymmetric noise and analyzing the effects of different loss functions.

---

## Methodology

### 1. Data Preparation
We introduce label noise to the CIFAR-10 dataset:  
- Symmetric noise: Labels are randomly flipped to any incorrect class with a noise rate $\( \eta \in [0.2, 0.8] \)$.  
- Asymmetric noise (BONUS experiment): Labels are flipped in a structured way (e.g., truck → automobile) with $\( \eta \in [0.1, 0.4] \)$.

---

### 2. Normalized Losses for Robustness
Models were trained using both vanilla and normalized loss functions:
- Vanilla losses: Cross-Entropy (CE), Focal Loss (FL).
- Normalized losses: Normalized Cross-Entropy (NCE), Normalized Focal Loss (NFL).

For each noise level, we compare the performance of these losses, evaluating their robustness using accuracy and loss curves.

---

### 3. APL Framework for Balancing Robustness and Performance
To address the underfitting issue of normalized losses, we implement the **Active-Passive Loss (APL) framework**, combining:
- **Active losses**: NCE, NFL (maximize correct class probability).
- **Passive losses**: MAE, RCE (minimize incorrect class probabilities).

Models trained with APL are compared against previous methods, demonstrating **improved noise tolerance** while maintaining competitive accuracy.

Other important features and methodology is added along with the code in the jupyter file in the form of markdown.

---

## **Results & Comparison**
- **Performance Metrics**: We evaluate test accuracy under different noise rates.  
- **Visualization**: We plot accuracy vs. noise level for all methods, highlighting the trade-offs between robustness and clean data performance.
- **Comparison**:
  - Normalized losses outperform vanilla losses in high-noise settings.
  - APL balances robustness and accuracy, mitigating underfitting while remaining noise-resistant.
 
- Train and Test accuracy and Loss for each dataset on every Loss function has been attached as well.

---

## **Conclusion**
This study shows that **normalized losses improve robustness**, but can suffer from underfitting. **APL mitigates this issue** by leveraging both active and passive losses, leading to a **better balance** between accuracy and robustness. The findings reinforce the importance of loss function design in **noisy-label learning**.

---

# Bonus: Asymmetric Noise Experiment
We extend the study to asymmetric noise $(\ \eta \in [0.1, 0.4] \)$, confirming that:
- Normalized losses remain robust but suffer from underfitting.
- APL still effectively balances performance and robustness.
- Asymmetric noise impacts performance less severely than symmetric noise.

---

## Future Work
Further research could explore:
- Combining APL with confidence-based sample selection.
- Extending experiments to larger datasets like CIFAR-100 and Imagenet.
- (Later addition) APL gave better results than SCEL on CIFAR 100 on moderate noise while SCEL overtook it and was able to give satisfying accuracy in similar or higher noise contexts
- Applying these techniques to semi-supervised learning or unsupervised learning scenarios, also a loss function that converges extremely fast would be ideal for PINNs.

---
