---
description: A brief comparison of the standard and modified D-H methods
---

# Comparison of Standard and Modified D-H Methods

To determine the relationship between the base and end-effector in terms of kinematics, coordinate frames are assigned to each link between the two. While these frames can be positioned arbitrarily, it is preferable to adopt a standardized approach for describing link parameters and joint configurations uniformly.

Each link is defined by two parameters, its length $$a_i$$ and twist $$\alpha_i$$. Joints are also described by two variables, the joint offset $$b_i$$, which is the distance between adjacent link frames along the joint axis, and the joint angle $$\theta_i$$, which represents the rotation of one link relative to the next about the joint axis.

The Denavit-Hartenberg (D-H) method is a widely used and well-established approach in the field of robotics. Although the original D-H method is still in use, a modified version is also commonly employed. While the modified D-H method provides clearer and more concise definitions, its adoption can cause confusion, particularly for those who use existing kinematics function libraries without understanding which D-H method the library is based on. Thus, this blog offers a comprehensive comparison of the standard and modified D-H methods to facilitate clarification.

## Standard D-H Method

The D-H method was first introduced in 1955. To differentiate it from the modified D-H method, the standard D-H method assigns the coordinate frame $$i$$ to the far end (distal end) of link $$i$$, as illustrated below:

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption><p>Illustration of standard D-H method</p></figcaption></figure>

The link and joint parameters are referred to as D-H parameters and are summarized in the following table. These definitions of D-H parameters remain consistent across all versions of the D-H method.

<table><thead><tr><th width="120"></th><th width="40"></th><th></th><th></th></tr></thead><tbody><tr><td>Joint angle</td><td><span class="math">\theta_i</span></td><td>The angle between <span class="math">x_{i-1}</span>and <span class="math">x_i</span> axes about <span class="math">z_{i-1}</span> axis</td><td>Joint variable</td></tr><tr><td>Joint offset</td><td><span class="math">b_i</span></td><td>The length between <span class="math">x_{i-1}</span>and <span class="math">x_i</span> axes along <span class="math">z_{i-1}</span> axis</td><td>Constant</td></tr><tr><td>Link length</td><td><span class="math">a_i</span></td><td><p>The length between <span class="math">z_{i-1}</span>and <span class="math">z_i</span></p><p> axes along <span class="math">x_{i-1}</span> axis</p></td><td>Constant</td></tr><tr><td>Link twist</td><td><span class="math">\alpha_i</span></td><td><p>The angle between <span class="math">z_{i-1}</span>and <span class="math">z_i</span></p><p> axes about <span class="math">x_{i-1}</span> axis</p></td><td>Constant</td></tr></tbody></table>

The transformation from coordinate frame $$i$$ to frame $$i-1$$ is expressed in terms of elementary rotations and translations as:

$$
\begin{eqnarray}
^{i-1}T_i(\theta_i, b_i, a_i, \alpha_i) &=& R_z(\theta_i)T_z(b_i)T_x(a_i)R_x(\alpha_i) \\
&=&
 \begin{bmatrix}
   \cos \theta_i & -\sin \theta_i \cos \alpha_i & \sin \theta_1 \sin \alpha_i  & a_i \cos \theta_i \\
   \sin \theta_i & \cos \theta_i \cos \alpha_i & - \cos \theta_i \sin \alpha_i & a_i \sin \theta_i \\
   0 & \sin \theta_i & \cos \alpha_i & b_i \\
   0 & 0 & 0 & 1 \\
\end{bmatrix}
\end{eqnarray}
$$

## Modified D-H Method

In 1986, Craig introduced the concept of modified D-H parameters, in which the link coordinate frames are attached to the near (proximal) end of each link, instead of the far end, as depicted in the figure below

<figure><img src="../.gitbook/assets/image (15).png" alt=""><figcaption><p>Illustration of modified D-H method</p></figcaption></figure>

$$
\begin{eqnarray}
^{i-1}T_i(\theta_i, b_i, a_{i-1}, \alpha_{i-1}) &=& R_x(\alpha_{i-1})T_x(a_{i-1})R_z(\theta_i)T_x(b_i) \\
&=&
 \begin{bmatrix}
   \cos \theta_i & -\sin \theta_i  & 0  & a_{i-1} \\
   \sin \theta_i \cos \alpha_{i-1} & \cos \theta_i \cos \alpha_{i-1} & - \sin \alpha_{i-1} & - \sin \alpha_{i-1} b_i \\
   \sin \theta_i \sin \alpha_{i-1} & \cos \theta_i \sin \alpha_{i-1} & \cos \alpha_{i-1} & \cos \alpha_{i-1} b_i \\
   0 & 0 & 0 & 1 \\
\end{bmatrix}
\end{eqnarray}
$$

## Impact of Rotation Order

The difference in homogeneous transformation matrices between the standard and modified D-H methods is due to the difference in rotation sequence, as rotations are not commutative.

The first three columns and rows of the transformation matrix represent the rotation matrix, which is solely determined by the sequence of joint angle $$\theta_i$$ and link twist $$\alpha_i$$ multiplications. As long as the two joint parameters, $$\theta_i$$ and $$b_i$$, and two link parameters, $$\alpha_i$$ and $$a_i$$, are kept in the same order in the transformation equation, both the rotation and translation matrices within the transformation matrix will remain the same.

When using the D-H method to construct kinematic parameters, either the standard or modified method may be used, but all parameters must consistently follow the same method from the start.
