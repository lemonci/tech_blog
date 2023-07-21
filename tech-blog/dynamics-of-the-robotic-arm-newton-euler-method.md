# Dynamics of the robotic arm: Newton - Euler Method

## Rigid body dynamics

We know that a robotic arm consists of a **joint** and a **linkage**: the joint is able to exert force in a specific direction on the linkage to which it is attached; the linkage is a **rigid body with a certain mass and size (so the inertia tensor is not negligible) that does not deform**. It follows that the essence of robotic arm dynamics is rigid body dynamics.

Rigid body dynamics is the study of the relationship between the state of motion of a rigid body and the external forces on it. This relationship is described by Euler's laws of motion. Euler's laws of motion are an extension of Newton's second law, known as $$F = ma$$; If Newton's law describes an abstract "object" or "particle", Euler's law extends it to an infinite number of particles assembled together to form a concrete rigid body with volume.

### Euler's first law of motion (Newton's law)

Euler's first law describes the linear motion of a rigid body in relation to the external forces applied to it.

First, the concept of linear momentum is introduced, which is equal to the mass of an object (particle) multiplied by the linear velocity. By Newton's second law($$\frac{d(mv)}{dt} = ma = F$$), **the rate of change of linear momentum is equal to the external force on the object**.

A rigid body can be viewed as a collection of countless particles, and the linear momentum of the rigid body is equal to the sum of the linear momentum of these particles. Due to the zero deformation property of the rigid body, an external force acting at any point on the rigid body is equivalent to a change in the linear momentum of all the particles. So,

$$\overrightarrow{F} = \Sigma_{i} \frac{dm_i \overrightarrow{v_i}}{dt}$$

The right hand side of the equation can be transformed into an integral over the mass of the rigid body:

$$\Sigma_im_i \overrightarrow{v_i} = \int_m \overrightarrow{v}(m)dm = \int_m \frac{d\overrightarrow{r}m}{dt}dm$$

where the vector $$\overrightarrow{r}$$ represents the position vector of each point on the rigid body in some frame of reference. If we use the center of mass as the original point:

$$\overrightarrow  r_{cm}= \frac {1}{m} \int_m \overrightarrow  r (m) dm$$. The subscript $$cm$$ represents for center of mass.

So,

$$\int_m \frac{d \overrightarrow r (m) }{dt} dm = m \centerdot \frac{d}{dt}\left( \frac{1}{m} \int_m \overrightarrow r (m) dm \right) =  m \centerdot \overrightarrow{v}_{cm}$$

Ultimately, the linear momentum of a rigid body is the product of the mass of the rigid body and the linear velocity of its center of mass. This conclusion shows that when considering the linear motion of a rigid body, it is sufficient to consider only the linear motion of the center of mass.

Euler's first law - the rate of change of the linear momentum of a rigid body is equal to the external force on it:

$$\overrightarrow{F} = \frac{d}{dt} m \overrightarrow{v}_{cm}$$

Obviously, the derivative of a linear velocity is a linear acceleration, so Euler's first law can also be written in this form below:

&#x20;$$\overrightarrow{F} = \frac{d}{dt} m \overrightarrow{v}_{cm}$$

### Euler's second law of motion

Euler's second law describes the relationship between the angular motion of a rigid body and the moment of force. When a force acts on a particle, it has a force arm with respect to another point O. The force arm multiplied by the force is the moment.

Moment is a vector, not a scalar, the direction points to the direction of rotation of the object. It is much clearer to memorize this relationship as a cross product! Also note the order of the cross multiplication.

$$\overrightarrow{N} = \overrightarrow{p} \times \overrightarrow{F}$$![](<../.gitbook/assets/image (13).png>)

Similar to linear momentum, we also introduce the concept of angular momentum: the angular momentum of a particle with respect to a fixed point is equal to the product of its mass and its angular velocity. Obviously, the linear velocity of the particle (with respect to this fixed point)  is equal to the cross product of the angular velocity (with respect to this fixed point), and the position vector from the fixed point to the particle, and : $$\overrightarrow{v} = \overrightarrow{p} \times \overrightarrow{\omega}$$

Then angular momentum is written as: $$\overrightarrow{M} = \overrightarrow{p} \times m \overrightarrow{v}$$

Starting from Newton's law $$\overrightarrow{F} = m \overrightarrow{a}$$, if we multiply it with the position vectors and use the the derivative of the vector cross product :

$$\begin{eqnarray}  \overrightarrow{F} &=& m \overrightarrow{a} \\ m \centerdot \overrightarrow{p} \times \frac{d}{dt} \overrightarrow{v} &=& m \centerdot \frac{d}{dt} \overrightarrow{p} \times \overrightarrow{v} - m \centerdot \frac{d}{dt} \overrightarrow{p} \times \overrightarrow{v} \\  \overrightarrow{p}   \times m \centerdot \frac{d}{dt} \overrightarrow{v} &=& m \centerdot \frac{d}{dt} \overrightarrow{p} \times \overrightarrow{v} - m \centerdot \overrightarrow{v} \times \overrightarrow{v} \\ \overrightarrow{p}   \times \overrightarrow{F} &=& m \centerdot \frac{d}{dt} \overrightarrow{p} \times \overrightarrow{v} \end{eqnarray}$$

We can then get a similar conclusion like the first law: **the rate of change of the angular momentum of an object is equal to the moment of force applied to it**, i.e.

$$\overrightarrow{N} = \frac{d}{dt} m \overrightarrow{\omega}$$

Now similarly, we need to generalize this conclusion to a rigid body made up of an infinite collection of particles; this part is slightly more problematic compared to the derivation of the first law.

First we have to derive an expression for the angular momentum of the rigid body, which, still considered as a collection of innumerable particles:

$$\overrightarrow{\Phi} = \Sigma_i m_i \overrightarrow{p_i} \times \overrightarrow{v_i} = \Sigma_i m_i \overrightarrow{p_i} \times （\overrightarrow{\omega} \times \overrightarrow{p_i})$$

Notice that we have replaced the linear velocity with a cross product of the angular velocity and the position vector, which is the same for every point on the rigid body, so we can take it outside the summation. Writing the above equation in volume integration form, we have:

$$\begin{eqnarray}   \overrightarrow{\Phi} &=& \int_V \overrightarrow{p} \times (\overrightarrow{\omega} \times \overrightarrow{p} ) \rho d v \\ \overrightarrow{\Phi} &=& \int_V \overrightarrow{p} \times ( - \overrightarrow{p} \times \overrightarrow{\omega} ) \rho d v \\  \overrightarrow{\Phi} &=& \left[ \int_V - \hat{p} \hat{p}\rho d v  \right] \overrightarrow{\omega}  \end{eqnarray}$$

p wedge in this context is called the cross product operator, which converts the cross product of two vectors into a matrix-by-vector operation. In our 3×1 vector case:

$$\begin{eqnarray}   \overrightarrow{p} \times \overrightarrow{\omega} &=& \left( \begin{vmatrix} p_y & p_z\\ \omega_y & \omega_z \end{vmatrix}, - \begin{vmatrix} p_x & p_z\\ \omega_x & \omega_z \end{vmatrix} , \begin{vmatrix} p_x & p_y\\ \omega_x & \omega_y \end{vmatrix} \right) \\  \overrightarrow{p} \times \overrightarrow{\omega} &=& \begin{bmatrix} 0 & -p_z & p_y\\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix}  \begin{bmatrix} \omega_x\\ \omega_y\\ \omega_z \end{bmatrix} \\  \hat{p} &=& \begin{bmatrix} 0 & -p_z & p_y\\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix} \end{eqnarray}$$

Here, we define the part in parentheses \[] in the previous integral equation as the inertia tensor $$I$$:

$$I = \left[ \int_V - \hat{p} \hat{p} \rho d v \right]$$.

Finally, the expression for the angular momentum of the rigid body is $$\Phi = I \overrightarrow{\omega}$$

Similar to force as the rate of change of linear momentum, the **torque** applied to a rigid body is defined as **the rate of change of the angular momentum of the rigid body**. To find the relationship between torque and angular velocity and angular acceleration, we can derive both sides of the above equation:

$$\begin{eqnarray}  \frac{d}{dt}\Phi &=& \frac{d}{dt} I \overrightarrow{\omega} \\ \overrightarrow{\tau} &=& I \frac{d\overrightarrow{\omega}}{dt} + \frac{dI}{dt} \overrightarrow{\omega} \end{eqnarray}$$

$$\frac {d \omega}{dt}$$ is obviously the angular acceleration of the rigid body, and what is $$\frac {I \omega}{dt}$$? It is the derivative of the tensor with respect to time, which is not necessarily equal to 0!

The tensor is the volume fraction of each point on a rigid body, after certain operations imposed on the position vectors with respect to a stationary frame of reference, and that these position vectors rotate as the rigid body rotates; similarly, the linear velocity of a point is equal to the forked product of the angular velocity and the position vectors:

$$\begin{eqnarray}  \overrightarrow{\tau} &=& I \overrightarrow{\alpha} + \overrightarrow{\omega} \times I \overrightarrow{\omega}  \end{eqnarray}$$

Summarize rigid body dynamics in a table:

|                  | Linear                    | Angular                                                                                                          |
| ---------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Inertia          | Mass $$m$$                | Tensor $$I$$                                                                                                     |
| Momentum         | $$mv$$                    | $$I \omega$$                                                                                                     |
| Force            | Force $$F$$               | Torque $$\tau$$                                                                                                  |
| Accelerate       | Linear acceleration $$a$$ | Angular acceleration $$\alpha$$                                                                                  |
| Eular's equation | $$F = ma$$                | $$\overrightarrow{\tau} = I \overrightarrow{\alpha} + \overrightarrow{\omega} \times I \overrightarrow{\omega}$$ |

## Deriving robotic arm dynamics using Newton's Euler method

After spending so much time on the Euler equations of motion for rigid body dynamics, we can finally take a look at the dynamics of a robotic arm. The ultimate goal of solving the dynamics of a robotic arm is to find out how much torque (for rotational joints) or force (for translational joints) should be applied by the actuators at each joint if we need to control the robot to follow a certain trajectory (don't forget that a trajectory is a function of position in respect to time).

To derive the dynamics of a robotic arm using Newton - Euler method, two steps are need:

1. **Forward propagation**: starting from the base, we calculate the velocity and acceleration of each linkage in turn, all the way to the end-effector; and then:
2. **Backward propagation**: starting from the external force on the end-effector, we calculate the torque/force of each joint in turn

The Newtonian Eulerian method for deriving robotic arm dynamics is a recursive algorithm.

### **Forward propagation**

First look at the angular velocity of the connecting rod, the angular velocity of each connecting rod is equal to the angular velocity of the last connecting rod plus the angular velocity of its joints (0 for translational joints); according to this we can get the formula for the transfer of angular velocity of the connecting rod. After derivating of both sides of the formula, we can further get the formula of angular acceleration:

$$\begin{eqnarray}   \omega_{i+1} &=& \omega_i + \dot{\theta_{i+1}}Z_{i+1} \\ \dot{\omega}_{i+1} &=& \dot{\omega}_i + \dot{\theta}_{i+1}(\omega_i \times Z_{i+1}) + \ddot{\theta}_{i+1} Z_{i+1}  \end{eqnarray}$$

Where $$\theta$$ is the joint position (angle of the rotated joint) and $$Z$$ is the rotation axis of the joint. Note that the above equation does not indicate the reference frame used for each vector. In practice, it is necessary to use a rotation matrix to map vectors from different reference frames to the same reference frame.

For linear velocity, the linear velocity of each link is equal to the sum of the linear velocity of the previous link (center of mass), the linear velocity caused by the rotation of the previous link, and the linear velocity of the translational joint. Derivation on both sides of the equation gives the transfer equation for linear acceleration.

$$\begin{eqnarray}  v_{i+1} &=& v_i + \omega_i \times p_{i+1} + \dot{d}_{i+1}Z_{i+1}\\ \ddot{v}_{i+1} &=& \ddot{v}_i + \ddot{\omega}_i \times p_{i+1} + \omega_i \times (\omega_i \times p_{i+1}) + \dot{d}_{i+1} {\omega}_i \times Z_{i+1} + \ddot{d}_{i+1} Z_{i+1} \end{eqnarray}$$

