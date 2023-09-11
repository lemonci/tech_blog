---
description: >-
  Even if you are not into robotics, the bicycle model will make your driver's
  license test easier. (I hope so :)
---

# Bicycle model for a car

The content is taken from the text book Corke, Peter I., Witold Jachimczyk, and Remo Pillat. _Robotics, vision and control: fundamental algorithms in MATLAB_. Vol. 73. Berlin: Springer, 2011.

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

I (and some labmates)had been stuck with the derivative of $$\alpha$$ for some time. Thanks for my co-supervisor, [Lionel Birglen](https://www.polymtl.ca/expertises/en/birglen-lionel), who points out that it is the derivative of $$\arctan$$ that confused us. So here it goes:

$$\begin{eqnarray} \frac{d} {dt} \left( \arctan \frac{\Delta y}{\Delta x} \right) &=& \frac{1}{1+{\left( \frac{\Delta x}{\Delta y}\right)}^2} \frac {d}{dt} \left( \frac{\Delta y}{\Delta x} \right) \\ &=& \frac{\Delta x^2}{\Delta x^2 + \Delta y^2} \left( \frac{ \dot{\Delta y} \Delta x - \Delta y \dot{\Delta x}} {\Delta x^2} \right) \\ &=& \frac {\dot{ \Delta y} \Delta x - \Delta y \dot {\Delta x}}{\rho^2}  \end{eqnarray}$$

Assume the goal position is fixed.

Remember, $$x$$ and $$y$$ are the x-axis and y-axis position of the robot, so we have $$\dot{x} = - \dot{\Delta x}$$and $$\dot{y} = - \dot{\Delta y}$$. Then:

&#x20;$$\begin{eqnarray} \frac{d} {dt} \left( \arctan \frac{\Delta y}{\Delta x} \right) &=& \frac{\Delta y \dot{x}-\Delta x \dot{y}}{\rho^2} \\ &=& \frac{v}{\rho^2} \left( \Delta y \cos \theta - \Delta x \sin \theta \right) \end{eqnarray}$$since $$\dot{x} = v \cos \theta$$ and $$\dot{y} = v \sin\theta$$

With $$\Delta x = \rho \cos(\alpha+\theta)$$and $$\Delta y = \rho \sin(\alpha+\theta)$$, comes:

$$\begin{eqnarray} \frac{d} {dt} \left( \arctan \left( \frac{\Delta y}{\Delta x} \right) \right) &=& \frac{v}{\rho^2} \cdot \rho (\sin (\alpha + \theta) \cos \theta - \cos(\alpha+\theta) \sin \theta) \\ &=& \frac{v \sin \alpha} {\rho} \end{eqnarray}$$



