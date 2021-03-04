---
title: How to load Tensorflow's SavedModel with multiple inputs/outputs in C++
tags: Tensorflow Keras C++
---

The [Keras/Tensorflow Python API](https://www.tensorflow.org/guide/keras/save_and_serialize) allows simple and easy model saving, loading and inference of trained models. But performing the same operations with C++ is somehow more complicated. This article will describe how to load a `SavedModel` with C++ for inference operations. 
<!--more-->

Using Python and _Tensorflow 2.X_, it has become super simple to _save_ and _load_ a model:
```python 
# Saving a Tensorflow model
model = ...  # Training
model.save('path/to/model')
```


```python
from tensorflow import keras

# loading a Tensorflow model
model = keras.models.load_model('path/to/model')

# perform inference
outputs = model(input)
```


#### Tensorflow Subclassed Model with multiple inputs and outputs
Even when the model architectures has multiple input and outputs the code remains very _pythonic_.


```python
# Tensorflow subclassed model with several inputs
class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()

  def call(self, inputs, training=False, mask=None):
    input1, input2 = inputs[0], inputs[1]
    ...
    ...
    ...
    return bbox_proposals, probabilities
```

We can simply call the model object directly to perform the inference. This will invoke the `MyModel`'s `call(inputs, training=False, mask=None)` function.

```python
model = keras.models.load_model('path/to/model')

# perform inference
bbox_proposals, probabilities = model([input1, input2])
```

#### Using the Tensorflow C++ API

However, using the Tensorflow C++ API to perform inference with a Keras' `SavedModel` is [not very well documented](https://www.tensorflow.org/guide/saved_model#load_a_savedmodel_in_c). The following snippets provide a small walk through for loading and inference of the Tensorflow Model with C++.


As described in the [official guide](https://www.tensorflow.org/guide/saved_model#load_a_savedmodel_in_c), it's recommended to use `tensorflow::SavedModelBundle` which contains the MetaGraphDef and the Tensorflow session.

```cpp
tensorflow::SessionOptions session_options_;
tensorflow::RunOptions run_options_;
tensorflow::SavedModelBundle model_;

auto status = tensorflow::LoadSavedModel(session_options_,
                                         run_options_,
                                         path_to_model_,
                                         {tensorflow::kSavedModelTagServe},
                                         &model_);
if (!status.ok()) {
    std::cerr << "Failed to load model: " << status;
return;
}
```
So far, so good. We successfully loaded the model. Now, comes the tricky part.
The model that we consider is still the subclassed model defined above with two inputs and two outputs. In order to feed the model successfully during the inference we need to know the __input node names__ and also the __output node names__ of the computational graph. Usually, a node name in Tensorflow can be defined with the parameter `name="node_name"`. But in our case the inputs nor the outputs where tagged like that. Furthermore, the documentation does not provide any information to solve that issue.


Hence, we can investigate the loaded model by using the signature map of the model.
```cpp
auto sig_map = model_.GetSignatures();
auto model_def = sig_map.at("serving_default");

printf("Model Signature");
for (auto const& p : sig_map) {
    printf("key: %s", p.first.c_str());
}

printf("Model Input Nodes");
for (auto const& p : model_def.inputs()) {
    printf("key: %s value: %s", p.first.c_str(), p.second.name().c_str());
}

printf("Model Output Nodes");
for (auto const& p : model_def.outputs()) {
    printf("key: %s value: %s", p.first.c_str(), p.second.name().c_str());
}
```
Which will print something like
```
Model Signature
key: __saved_model_init_op
key: serving_default
Model Input Nodes
key: input_1 value: serving_default_input_1:0
key: input_2 value: serving_default_input_2:0
Model Output Nodes
key: output_1 value: StatefulPartitionedCall:0
key: output_2 value: StatefulPartitionedCall:1
```

Great, our input nodes are called 
- `serving_default_input_1:0`
- `serving_default_input_2:0`

and the output nodes
- `StatefulPartitionedCall:0`
- `StatefulPartitionedCall:1`.


We can use that information to perform the inference session by getting the node names:
```cpp
input_name_1 = model_def.inputs().at("input_1").name();
input_name_2 = model_def.inputs().at("input_2").name();

output_name_bbox_proposals = model_def.outputs().at("output_1").name();
output_name_probabilities = model_def.outputs().at("output_2").name();
```

And then finally run the model within the session.
```cpp
std::vector<TFTensor> inputTensor_1;
std::vector<TFTensor> inputTensor_2;
std::vector<TFTensor> bbox_output;

// fill the input tensors with data

tensorflow::Status status;
status = model_.session->Run({ {input_name_1, inputTensor_1},
                               {input_name_2, inputTensor_2} },
                               {output_name_bbox_proposals}, {}, &bbox_output);
if (!status.ok()) {
    std::cerr << "Inference failed: " << status;
    return;
}
```

Note, that on a GPU device we can use similar to the Python API the `set_allow_growth` variable for proper GPU RAM allocation.
```cpp
session_options_.config.mutable_gpu_options()->set_allow_growth(true);
```

#### Troubleshooting

In case you are encountering the following error during runtime, 

```
undefined symbol: _ZNK6google8protobuf8internal12MapFieldBase24SyncMapWithRepeatedFieldEv
```

you need to make sure that you link the C++ code against Google's __Protobuf library__. These libs can be installed with 

```bash
sudo apt install -y libprotobuf-dev
sudo apt install -y protobuf-compiler
```

#### Wrap-up

- We should load a Keras' `SavedModel` with `tensorflow::LoadSavedModel` in C++
- The names of input and output nodes of a subclassed Tensorflow model are not always uniquely defined 
- We can look up the node names by examining the signature map of the model
- The inference of the model is then performed by call the model's session 