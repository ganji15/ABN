# ABN
ABN

batch norm:
在训练的时候会计算average的mean和std
但是训练时使用的是基于batch的mean和std，只有测试的时候使用average的mean和std，训练时不使用基于batch的mean和std的原因是？  看不懂作者解释，个人理解是如果使用average会使得训练的分布极其相似，拉近了各个独立batch的距离。而使用基于batch的统计值可以增加更新的多样性。如果minibatch sgd和full batch sgd对收敛的影响。

It is natural to ask whether we could simply use the moving averages μ, σ to perform the normalization during training, since this would remove the dependence of the normalized activations on the other example in the mini- batch. This, however, has been observed to lead to the model blowing up. As argued in [6], such use of mov- ing averages would cause the gradient optimization and the normalization to counteract each other. For example, the gradient step may increase a bias or scale the convolutional weights, in spite of the fact that the normalization would cancel the effect of these changes on the loss. This would result in unbounded growth of model parameters without actually improving the loss. It is thus crucial to use the minibatch moments, and to backpropagate through them.







normalization propagation:不是adaptive的normalization，用权重来做normalization以此消除依赖于batch的弊端，缺点是改变了模型

virtual batch normalization
用独立的batch(所以叫虚拟batch)来作为计算mean和std的batch，感觉初衷是希望计算统计值的batch和过网络的batch越不同越好。
Batch normalization greatly improves optimization of neural networks, and was shown to be highly effective for DCGANs [3]. However, it causes the output of a neural network for an input example x to be highly dependent on several other inputs x? in the same minibatch. To avoid this problem we introduce virtual batch normalization (VBN), in which each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training, and on x itself. The reference batch is normalized using only its own statistics. VBN is computationally expensive because it requires running forward propagation on two minibatches of data, so we use it only in the generator network.

batch renorm
相比bn，bn训练使用基于batch的统计值，测试使用基于average的统计值
而batch renorm在训练和测试都使用基于average的统计值，只不过通过 基于batch的统计值来表达基于average的统计值，具体做法相当于在原本基于batch的normalized输出之后做了一个per-dimension的线性变换，但线性变化拉伸量r和偏移值d在训练时被当作是constant(对于给定minibatch)。
具体r= sigma{ba}/sigma{avg}
d=(u{b}-u/)sigma{avg}
可以从值看出来，当minibatch与average的值一致时，该线性变换是identity mapping，因为bn此时已经没有缺陷；
否则，bn将被r和d再修正，故称之为batch renorm
修正可以设置rmax和dmax来截断，以免修正过度


设置了一个gradient clip的门槛



weight-normalization


streaming normalization
总结了基本的normalization 的方式，提出了参考前后batch的streaming normalization的方式