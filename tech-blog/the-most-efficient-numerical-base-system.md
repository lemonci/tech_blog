---
description: An optimization problem with constraints
---

# The most efficient numerical base system

Credit per [StackExchange](https://math.stackexchange.com/a/1362408/1148626).

Assume there are $$V$$ independent states of information. Then we can represent approximately $$\frac{V}{N}$$ digits in base $$N$$.

The amount of information we can represent is $$I=N^{\frac{V}{N}}$$.

The value of $$N$$that maximizes $$I$$ (either where the derivative is $$0$$(if the second derivative is negative) or at infinity (if the second derivative is positive)) is the most "efficient" base.

So we take the natural log: $$\ln(I)=\frac{V}{N}\ln(N)$$

And take the derivative to $$N$$: $$(\ln(I))^′=V\frac{(1−\ln(N))}{N^2}$$.

We then set $$(\ln(I))′=0$$. Solving, $$N=e$$.

Take the second derivative: $$(\ln(I))′′=V\frac{(2\ln(N)−3)}{N^3}$$

When $$N=e$$, $$(\ln(I))^{′′}=−\frac{V}{e^3}$$ which is negative (recall that $$V$$ is positive), so $$I$$ reaches its maximum when $$N=e$$.
