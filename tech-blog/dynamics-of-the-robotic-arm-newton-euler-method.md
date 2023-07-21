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
