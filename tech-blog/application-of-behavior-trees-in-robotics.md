---
description: >-
  Task scheduling for robots is usually implemented using finite state machines,
  and behaviour trees are another way of implementing task scheduling for
  robots.
---

# Application of Behavior Trees in Robotics

The Behavior Tree (BT) is a structure for task switching, similar to the Finite State Machine (FSM). . They are often used in robotics and game AI scenarios where the actions of objects are often edited in advance, but there is uncertainty about what action to use at what time or place. Therefore it is necessary to use BTs or FSMs to implement decision making tasks for these intelligences.

In contrast to the earlier FSM, BTs have the following advantages:

* **Modularity**: BTs allow action elements to be added or removed at will. We can combine complex behaviors, including the whole tree as a sub-branch of a larger tree.
* **Semantic graphicality**: BTs can be easily converted into graphical processes, whereas state machines are relatively difficult.&#x20;
* **Greater expressiveness**:  inherent logic processing tools such as sequences, loops, breaks etc.

## Basic concepts of behavior trees

During runtime, a signal called `tick` is sent from the root of the tree and tranmits through the tree until it reaches a leaf node. Any **TreeNode** that receives the signal will execute its callback. This callback must return the following three states: **success**, **failure**, **running**

Running means that the action takes longer to return the result.

**Leaf node**s are those tree nodes that do not have any children, they are the actual commands, i.e. the nodes where the BTs interact with the rest of the robot's system. The most common leaf nodes are the action nodes. For example:

![](../.gitbook/assets/v2-d9b3c22d11607509cf8e843a0cc3c26f\_b.gif)

The **Sequence** is the most basic control node in the behavior tree and will execute all its child nodes in turn. It will return **success** if all the children nodes return **success**.

The first **tick** sets the **Sequence** node to '**running**' (orange). The second tick executes the leftmost child node (**OpenDoor**) for the sequential node and gets its success signal back. The third and fourth ticks then execute the other two child nodes of the sequence node (**Walk** and **CloseDoor**) respectively. The sequence node also switches from **running** to **success** after obtaining the **success** signal from the last child node.

## Node type

* TreeNode
  * Decorator node
  * Control node
  *   Leaf node

      * Condition node
      * Action node

      <figure><img src="../.gitbook/assets/image (10).png" alt=""><figcaption><p>Node type</p></figcaption></figure>

### Decorator node

Depending on the type of Decorator, the goal of this node could be either:

* to transform the result it received from the child.
* to halt the execution of the child.
* to repeat ticking the child, depending on the type of Decorator.

#### Inverter node

If a child node returns **success**, it returns **failure**. If the child node returns **running**, this node also returns **running**.

#### Force success node

If a child node returns **running**, then this node also returns **running**. Otherwise, it will always return **success**.

#### Force failure node

If the child node returns **running**, then this node also returns **running**. Otherwise, it always returns **failure**.

#### Repeat node

If the child node returns **success**, the child node is executed again, up to a maximum of N times. If the child node returns a **failure**, the loop is broken and the node returns a **failure**. If the child node returns **running**, then this node also returns **running**.

#### Retry node

The opposite of a repeat node. If the child node returns **success**, the loop is interrupted. If the child node returns a **failure**, the return failure executes the child node again, up to N times. If the child node returns **running**, this node also returns **running**.

### Control node

The most common type of control node is the **sequence node**, just shown in the example above. Control nodes also include variations of sequential nodes, such as `ReactiveSequence` and `SequenceWithMemory`.

#### `ReactiveSequence` node

The following is an example of a commonly used reaction order node in games. The agent will approach an enemy when it is within view. This section of the behavior tree calls the `IsEnemyVisible` condition repeatedly during the runtime. If the condition is true, the `ApproachEnemy` action will be executed, but the enemy may escape from the view of the agent during the action, so this `ApproachEnemy` action is asynchronous, i.e. the agent moves a little closer to the enemy's position each time until it is completely close to the the enemy, before returning a **success** state. Until then, the `ApproachEnemy` will remain in the **running** state.

![](<../.gitbook/assets/image (6).png>)

#### `SequenceWithMemory` node

In the following example, the robot receives an instruction to go to locations A, B and C. If the robot experiences **failure** when after executed `GoTo(B)`, `GoTo(A)` will not receive a `tick` signal again.

![](<../.gitbook/assets/image (1).png>)

#### Fallback node

The purpose of a fallback node (also known as a selector) is to try out different strategies until we find one that "works". That is, what to do next if a child node returns **failure**. The framework currently offers two types of nodes: fallback node and reactive fallback node

They share the following rules:

If the child node returns a **failure**, the next child node is executed. If the child node returns a **success**, no further child nodes are executed and the fallback returns a success. If all children return a **failure**, the fallback also returns a **failure**.

<figure><img src="../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

The process above :

* Is the door open?
* If not, try to open the door.
* Otherwise, if you have the key, unlock and open the door.
* Otherwise, smash the door.
* If either of these actions is successful, enter the room.

#### Reactive fallback node

The asynchronous behavior in a reactive fallback is interrupted when its child node changes from a **failure** to a **success**. As in the example below, the agent will `sleep` continuously for 8 hours after satisfying the `rest` condition.

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption></figcaption></figure>
