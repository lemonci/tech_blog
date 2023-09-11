---
description: >-
  Even if you are not into robotics, the bicycle model will make your driver's
  license easier. (I hope so :)
---

# Bicycle model for a car

The content is taken from the text book Corke, Peter I., Witold Jachimczyk, and Remo Pillat. _Robotics, vision and control: fundamental algorithms in MATLAB_. Vol. 73. Berlin: Springer, 2011.

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

I (and some labmates)had been stuck with the derivative of $$\alpha$$ for some time. Thanks for my co-supervisor, [Lionel Birglen](https://www.polymtl.ca/expertises/en/birglen-lionel), who points out that it is the derivative of $$\arctan$$ that confused us. So here it goes:

$$\begin{eqnarray} \frac{d} {dt} \left( \arctan \frac{\Delta y}{\Delta x} \right) &=& \frac{1}{1+{\left( \frac{\Delta x}{\Delta y}\right)}^2} \frac {d}{dt} \left( \frac{\Delta y}{\Delta x} \right) \\ &=& \frac{1}{1+{\left( \frac{\Delta x}{\Delta y}\right)}^2} \left( \frac{ \Delta \dot{y} \Delta x - \Delta y \Delta \dot{x}} {\Delta x^2} \right) \end{eqnarray}$$





