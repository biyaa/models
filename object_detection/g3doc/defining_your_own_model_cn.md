# 来，建立你想要一个新的模型！
在这个章节，为了定义我们使用物品定位模型，我们会讨论一些关于这方面的抽象。
如果你想使用TF物品定位API来定义一个新模型架构，这节内容也会作你需要编辑的
文件内容的来让你的新的模型能够有效工作的高阶向导。
## DetectionModels (`object_detection/core/model.py`)
为了能够训练、评估和能用我们提供的二进制包来导出为服务，所有使用TF物品定位API
必需实现`DetectionModel` 接口 ( `object_detection/core/model.py`有完整定义).
特别说明，这些模型负责实现这5个功能：
* `preprocess`: 输入图片做必要前期处理（如：缩放、移位、变形）
* `predict`: 产生原始的预测张量，这些张量可能传递给LOSS函数或进行后期处理
* `postprocess`: 转换输出张量为最终物品定位的结果
* `loss`: 计算相对于提供真实数据的LOSS值
* `restore`: 调出模型保存点数据到TF图中
在训练阶段，给定的一个 `DetectionModel`，每批图片会顺序通过下面的功能，最后会计算出
LOSS值，这个LOSS值又能够通过SGD优化：
```
inputs (images tensor) -> preprocess -> predict -> loss -> outputs (loss tensor)
```
在评估阶段，每批图片会顺序通过下面的功能，会产生一组物品定位预测：
```
inputs (images tensor) -> preprocess -> predict -> postprocess ->
  outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)
```
为了方便理解，做出以下约定：
* `DetectionModel` ：不应假定输入大小和高宽比 --- 它们会负责做必要的缩放和变形
* 输出的物品分类的总是在整数范围，如：`[0, num_classes)`。
 任何整数映射成能够理解的语言标签都在这个API进行。 我们不会明确的发出一个背景分类
---所以0是第一个非背景类别，任何预测逻辑和删除暗示的背景类别，必须实现者自己内部
处理这种情况。
* 定位检测出坐标格式为：`[y_min, x_min, y_max, x_max]` ，并且是image的相对坐标。
* 我们任何种类分数不做概率性解释假定---唯一重要的只是他们相对顺序。所以你后期处理实现
 时，可以输出对数、可能性或校对可能性或者其它的东西
 
## 定义一个新的 Faster R-CNN 或 SSD 特征提取器
在大部分情况下，你可能不需要实现`DetectionModel`的这个部分 --- 你实际可能是通过Faster R-CNN 或 SSD
元架构来创建一个新的特征提取器。（我们认为元架构作通过使用`DetectionModel`抽象来定义整个模型家族更恰当。）

注：为了能理解接下来的讨论, 我们推荐先熟悉这个论文 [Faster R-CNN](https://arxiv.org/abs/1506.01497) 。

让我们现成假定你已经发明了一个新网络架构（假设叫“InceptionV100”）来做分类，想看一下InceptionV100作为
一个物体检测特征提取器（假定和 Faster R-CNN组合）的效果如何。 这个SSD模型的过程相似，但这里我们讨论 Faster R-CNN。

为了使用 InceptionV100, 我们必须定义一个新的`FasterRCNNFeatureExtractor` 并且作为 `FasterRCNNMetaArch`构造者
的输入。  见`object_detection/meta_architectures/faster_rcnn_meta_arch.py` 分别定义了 `FasterRCNNFeatureExtractor` 
和`FasterRCNNMetaArch`。
一个 `FasterRCNNFeatureExtractor` 必须定义功能：
* `preprocess`: 图片前期处理。
* `_extract_proposal_features`: 提取第一阶段的区域建议网络的 (RPN) 特征。
* `_extract_box_classifier_features`: 提取第二阶段矩阵框分类特征。
* `restore_from_classification_checkpoint_fn`: 调取保存点到TF图的功能。

见 `object_detection/models/faster_rcnn_resnet_v1_feature_extractor.py`的定义例子，一些注意事项
* 我们一般初始化特征提取器的权重使用的tf-slim  [Slim Resnet-101 classification checkpoint](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)，当我们训练时，图片的前期处理
这个保存点是减去每张输入图片通道的平均值。 所有我们的实现前期处理的功能是复制的这个，做同样的减去平均值的处理。
* 在SLIM中，完整的RESNET分类网络被切分成两个部分 --- “resnet block” 放在 `_extract_proposal_features` 功能中
final block 定义在 `_extract_box_classifier_features function`。  通常情况下，一些实验需要决定在优化层之上，切分
你的特征提取器到这两部分给 Faster R-CNN。

## 在配置中注册你的模型
假定你的模型不需要非标准配置，在你的配置文件中，可简单的修改“feature_extractor.type” 字段来指向新的特征提取器。为了
让我们的API知道怎么理解这个的类型，你首先必须注册你的新的特征提取器到模型建造者(`object_detection/builders/model_builder.py`)
它的功能是为了根据配置原理创建模型。

注册非常简单 --- 就加一个个你已经定义的SSD或 Faster R-CNN类中的新特征提取器在文件`object_detection/builders/model_builder.py`。
推荐加一个测试在`object_detection/builders/model_builder_test.py`，确认分析proto符合预期。


## 让新模型转起来
在注册好后，你准备去跑你的模型。最后的一些提示：

* 为了节约调试时间，首先运行本地的配置（包括训练和评估）
* 做一个适合模型的学习率的梳理。
* 一个小但经常用的重要细节：你可能需要禁用BN训练（哪是因为，调出BN参数分类保存点，但不会梯度下降中不会更新它们）。


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# So you want to create a new model!

In this section, we discuss some of the abstractions that we use
for defining detection models. If you would like to define a new model
architecture for detection and use it in the Tensorflow Detection API,
then this section should also serve as a high level guide to the files that you
will need to edit to get your new model working.

## DetectionModels (`object_detection/core/model.py`)

In order to be trained, evaluated, and exported for serving  using our
provided binaries, all models under the Tensorflow Object Detection API must
implement the `DetectionModel` interface (see the full definition in `object_detection/core/model.py`).  In particular,
each of these models are responsible for implementing 5 functions:

* `preprocess`: Run any preprocessing (e.g., scaling/shifting/reshaping) of
  input values that is necessary prior to running the detector on an input
  image.
* `predict`: Produce “raw” prediction tensors that can be passed to loss or
  postprocess functions.
* `postprocess`: Convert predicted output tensors to final detections.
* `loss`: Compute scalar loss tensors with respect to provided groundtruth.
* `restore`: Load a checkpoint into the Tensorflow graph.

Given a `DetectionModel` at training time, we pass each image batch through
the following sequence of functions to compute a loss which can be optimized via
SGD:

```
inputs (images tensor) -> preprocess -> predict -> loss -> outputs (loss tensor)
```

And at eval time, we pass each image batch through the following sequence of
functions to produce a set of detections:

```
inputs (images tensor) -> preprocess -> predict -> postprocess ->
  outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)
```

Some conventions to be aware of:

* `DetectionModel`s should make no assumptions about the input size or aspect
  ratio --- they are responsible for doing any resize/reshaping necessary
  (see docstring for the `preprocess` function).
* Output classes are always integers in the range `[0, num_classes)`.
  Any mapping of these integers to semantic labels is to be handled outside
  of this class.  We never explicitly emit a “background class” --- thus 0 is
  the first non-background class and any logic of predicting and removing
  implicit background classes must be handled internally by the implementation.
* Detected boxes are to be interpreted as being in
  `[y_min, x_min, y_max, x_max]` format and normalized relative to the
  image window.
* We do not specifically assume any kind of probabilistic interpretation of the
  scores --- the only important thing is their relative ordering. Thus
  implementations of the postprocess function are free to output logits,
  probabilities, calibrated probabilities, or anything else.

## Defining a new Faster R-CNN or SSD Feature Extractor

In most cases, you probably will not implement a `DetectionModel` from scratch
--- instead you might create a new feature extractor to be used by one of the
SSD or Faster R-CNN meta-architectures.  (We think of meta-architectures as
classes that define entire families of models using the `DetectionModel`
abstraction).

Note: For the following discussion to make sense, we recommend first becoming
familiar with the [Faster R-CNN](https://arxiv.org/abs/1506.01497) paper.

Let’s now imagine that you have invented a brand new network architecture
(say, “InceptionV100”) for classification and want to see how InceptionV100
would behave as a feature extractor for detection (say, with Faster R-CNN).
A similar procedure would hold for SSD models, but we’ll discuss Faster R-CNN.

To use InceptionV100, we will have to define a new
`FasterRCNNFeatureExtractor` and pass it to our `FasterRCNNMetaArch`
constructor as input.  See
`object_detection/meta_architectures/faster_rcnn_meta_arch.py` for definitions
of `FasterRCNNFeatureExtractor` and `FasterRCNNMetaArch`, respectively.
A `FasterRCNNFeatureExtractor` must define a few
functions:

* `preprocess`: Run any preprocessing of input values that is necessary prior
  to running the detector on an input image.
* `_extract_proposal_features`: Extract first stage Region Proposal Network
  (RPN) features.
* `_extract_box_classifier_features`: Extract second stage Box Classifier
  features.
* `restore_from_classification_checkpoint_fn`: Load a checkpoint into the
  Tensorflow graph.

See the `object_detection/models/faster_rcnn_resnet_v1_feature_extractor.py`
definition as one example. Some remarks:

* We typically initialize the weights of this feature extractor
  using those from the
  [Slim Resnet-101 classification checkpoint](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models),
  and we know
  that images were preprocessed when training this checkpoint
  by subtracting a channel mean from each input
  image.  Thus, we implement the preprocess function to replicate the same
  channel mean subtraction behavior.
* The “full” resnet classification network defined in slim is cut into two
  parts --- all but the last “resnet block” is put into the
  `_extract_proposal_features` function and the final block is separately
  defined in the `_extract_box_classifier_features function`.  In general,
  some experimentation may be required to decide on an optimal layer at
  which to “cut” your feature extractor into these two pieces for Faster R-CNN.

## Register your model for configuration

Assuming that your new feature extractor does not require nonstandard
configuration, you will want to ideally be able to simply change the
“feature_extractor.type” fields in your configuration protos to point to a
new feature extractor.  In order for our API to know how to understand this
new type though, you will first have to register your new feature
extractor with the model builder (`object_detection/builders/model_builder.py`),
whose job is to create models from config protos..

Registration is simple --- just add a pointer to the new Feature Extractor
class that you have defined in one of the SSD or Faster R-CNN Feature
Extractor Class maps at the top of the
`object_detection/builders/model_builder.py` file.
We recommend adding a test in `object_detection/builders/model_builder_test.py`
to make sure that parsing your proto will work as expected.

## Taking your new model for a spin

After registration you are ready to go with your model!  Some final tips:

* To save time debugging, try running your configuration file locally first
  (both training and evaluation).
* Do a sweep of learning rates to figure out which learning rate is best
  for your model.
* A small but often important detail: you may find it necessary to disable
  batchnorm training (that is, load the batch norm parameters from the
  classification checkpoint, but do not update them during gradient descent).
