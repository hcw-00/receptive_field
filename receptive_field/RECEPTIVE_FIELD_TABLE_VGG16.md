# Pre-computed receptive field parameters

## Table with results

The table below presents the receptive field parameters and cost (in terms of
floating point operations &mdash; FLOPs) for several popular convolutional
neural networks and their end-points. These are computed using the models from
the
[TF-Slim repository](https://github.com/tensorflow/models/tree/master/research/slim),
by using the
[rf_benchmark script](https://github.com/google-research/receptive_field/blob/master/receptive_field/python/util/examples/rf_benchmark.py).

Questions? See the [FAQ](#faq).

CNN                            | resolution | end-point            | FLOPs (Billion) | RF   | effective stride | effective padding
:----------------------------: | :--------: | :------------------: | :-------------: | :--: | :--------------: | :---------------:
vgg_16                         | 224        | vgg_16/conv1/conv1_1 | 0.177           | 3    | 1                | 1
vgg_16                         | 224        | vgg_16/pool1         | 3.882           | 6    | 2                | 2
vgg_16                         | 224        | vgg_16/conv2/conv2_1 | 5.734           | 10   | 2                | 4
vgg_16                         | 224        | vgg_16/pool2         | 9.436           | 16   | 4                | 6
vgg_16                         | 224        | vgg_16/conv3/conv3_1 | 11.287          | 24   | 4                | 10
vgg_16                         | 224        | vgg_16/conv3/conv3_2 | 14.987          | 32   | 4                | 14
vgg_16                         | 224        | vgg_16/pool3         | 18.688          | 44   | 8                | 18
vgg_16                         | 224        | vgg_16/conv4/conv4_1 | 20.538          | 60   | 8                | 26
vgg_16                         | 224        | vgg_16/conv4/conv4_2 | 24.238          | 76   | 8                | 34
vgg_16                         | 224        | vgg_16/conv4/conv4_3 | added by hcw    | 92   | 8                | 42
vgg_16                         | 224        | vgg_16/pool4         | 27.938          | 100  | 16               | 42
vgg_16                         | 224        | vgg_16/conv5/conv5_1 | 28.863          | 132  | 16               | 58
vgg_16                         | 224        | vgg_16/conv5/conv5_2 | 29.788          | 164  | 16               | 74
vgg_16                         | 224        | vgg_16/pool5         | 30.713          | 212  | 32               | 90

## FAQ

### What does a resolution of 'None' mean?

In this case, the input resolution is undefined. For most models, the receptive
field parameters can be computed even without knowing the input resolution. The
number of FLOPs cannot be computed in this case.

### For some networks, effective_padding shows as 'None' (eg, for Inception_v2 or Mobilenet_v1 when input size is not specified). Why is that?

This means that the padding for these networks depends on the input size. So,
unless we know exactly the input image dimensionality to be used, it is not
possible to determine the padding applied at the different layers. Look at the
other entries where the input size is fixed; for those cases, effective_padding
is not None.

This happens due to Tensorflow's implementation of the 'SAME' padding mode,
which may depend on the input feature map size to a given layer. For background
on this, see
[these notes from the TF documentation](https://www.tensorflow.org/versions/master/api_guides/python/nn#Notes_on_SAME_Convolution_Padding).

Also, note that in this case the program is not able to check if the network is
aligned (ie, it could be that the different paths from input to output have
receptive fields which are not consistently centered at the same position in the
input image).

So you should be aware that such networks might not be aligned -- the program
has no way of checking it when the padding cannot be determined.

### The receptive field parameters for network X seem different from what I expected... maybe your calculation is incorrect?

First, note that the results presented here are based on the tensorflow
implementations from the
[TF-Slim model library](https://github.com/tensorflow/models/tree/master/research/slim).
So, it is possible that due to some implementation details the RF parameters are
different.

One common case of confusion is the TF-Slim Resnet implementation, which applies
stride in the last residual unit of each block, instead of at the input
activations in the first residual unit of each block (which is what is described
in the Resnet paper) -- see
[this comment](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py#L30).
This makes the stride with respect to each convolution block potentially
different. In this case, though, note that a
[flag](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L150)
may be used to recover the original striding convention.

Second, it could be that we have a bug somewhere. While we include
[many tests](https://github.com/google-research/receptive_field/blob/master/receptive_field/python/util/receptive_field_test.py)
in our library, it is always possible that we missed something. If you suspect
this is the case, please file a GitHub issue
[here](https://github.com/google-research/receptive_field/issues).

### The number of FLOPs for network X seem different from what I expected... maybe your calculation is incorrect?

First, note that the results presented here are based on the tensorflow
implementations from the
[TF-Slim model library](https://github.com/tensorflow/models/tree/master/research/slim).
So, it is possible that due to some implementation details the number of FLOPs
is different.

Second, one common confusion arises since some papers refer to FLOPs as the
number of Multiply-Add operations; in other words, some papers count a
Multiply-Add as one floating point operation while others count as two. Here, we
follow the `tensorflow.profiler` convention and count a Multiply-Add as two
operations. One noticeable counter-example is the
[ResNet paper](https://arxiv.org/abs/1512.03385), where the FLOPs mentioned in
Table 1 therein actually mean the number of Multiply-Add's (see Section 3.3 in
their paper). So there is roughly a factor of two between their paper and our
numbers.

Finally, we rely on `tensorflow.profiler` for estimating the number of floating
point operations. It could be that they have a bug somewhere, or that we are
using their library incorrectly, or that we simply have a bug somewhere else. If
you suspect this is the case, please file a GitHub issue
[here](https://github.com/google-research/receptive_field/issues)).
