# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Custom layers are layers that are not included in the list of known layers supported by OpenVino. These layers can be added to OpenVino inference engine as customer layers. 

To add customer layers, two customer layer extensions are needed: Custom Layer Extractor and Custom Layer Operation.

Some of the potential reasons for handling custom layers are: some layers in a neural network model might  not be in the openVINO Model Optimizer supported layers list. They are layers that are not natively supported by the openvino Inference engine. These layers can be added to the Inference Engine as custom layers. The custom layer can therefore be defined as any model layer that is not natively supported by the model framework. 

## Comparing Model Performance

To compare model performance, I measured key performance metrics such as model size, inferencing speed and accuracy for the orifinal TF model and the post-optimization model.

A. Model optimization:
The first step is to run OpenVino model optimization in Udacity workspace. Steps are as follows:

1. Use OpenVino download script to download public model. The specific one I selected is faster_rcnn_inception_v2_coco. The sequence of commands is as follows:

> cd /home/workspace
> mkdir models
> cd ./models
> /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name faster_rcnn_inception_v2_coco -o .

2. Use OpenVino model optimizer script to convert the TensorFlow based model to IR form:

> /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ./public/faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ./public/faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

The model sizes can be compared by the model sizes before and after optimization operation.
 
After optimization, two key files for OpenVino IR are generated: frozen_inference_graph.bin and frozen_inference_graph.xml.

B. Capture the performance metrics:
Next step is to use the original Tensorflow model to measure its inference performance. Evaluation of the original Tensorflow model is based on this link which is discussed in Udacity forum: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb

The last step is basically implementing the people counter using OpenVino project itself. Key performance metrics like inferencing time per frame, accuracy of inference (ration of number of frames where people are detected vs the total number of frames with people inside) are calculated while running inferencing. 

C. Performance comparison: 
Model sizes - The model size comparison is based on teh file sizes before and after optimization. There appear to be significant reduction in model size: faster_rcnn_inception_v2_coco_2018_01_28 was reduced from the original ~170MB total size to ~55MB after optimization.

Model accuracy - This metric is calculated by counting the total number of people detected and the total number of people in frame, then calculate the ratio. For probability threshold of 0.6, both the original and post-optimization model yield accuracy values at about 94%. There's pratically no difference in accuracy between th eoriginal and post-optimization model. 

Inference time - By averaging the inference time per frame we got the for the original model the inferencing time is about 930ms per frame. For the post-optimization model we got about 940ms per frame. There is no major differences between thetwo models.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are: street traffic flow monitoring, retail store monitoring, building capacity monitoring (such as in restaurant, concert, meetings) as well as home security detection. 

Each of these use cases would be useful because the app would be able to detect human presence and how many people have entered/exited the premise. This is even more relevant in the COVID world where social distancing is very important.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:


- Lighting: It would have huge impact on detection results. Good lighting is important for proper detection and poor lighting would make model hard or even unable to detect humans.

- Model accuracy: naturally higher the accuracy the better the detection rate. However, as with lighting, model accuracy is only based on training materials and the inference results can be affected by other factors as well.

- Camera focal/length: these would determine whether the video delivered to model for inferencing is good to begin with. Without proper original images even the best model can do do too much.


Results:

For this project I use udacity workspace. The OpenVino version I used is 2019 R3. So some of the OpenVino updates mentioned in the class apply here. The model I used is the public Tensorflow model faster_rcnn_inception_v2_coco which is downloaded using OpenVino model downloader. After converting to IR form I was able to use it for the app and it appears to work well. The app correctly detects when a person appears in video, calculates the time of his/her stay when he/she exits, and send the statistics over to the browser to be displayed correctly. To deal with false negatives I use a time window to determine whether a missing person inferencing result is caused by real exit or just a false negative.  
An inference video with bounding box is stored in the "output" directory.

