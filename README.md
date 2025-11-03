Implementing a Carlini-Wagner black-box adversarial attack 
https://arxiv.org/pdf/1608.04644

Goal: optimize a set of parameters to model adversarial noise. Adding modeled noise to the original image results in misclassification by pre-trained resnet18 to a target class t.

L2 attack was used as described in the paper.  

minimize_w ||1/2(tanh(w) + 1) - x||²₂ + c · f(1/2(tanh(w) + 1))

where f is defined as:

f(x') = max(max{Z(x')ᵢ : i ≠ t} - Z(x')ₜ, -κ)

delta = 1/2(tahn(w) + 1) - x
x + delta = 1/2(tahn(w) + 1), thus 0 <= x + delta <= 1