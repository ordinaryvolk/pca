# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves adding two necessary custom layer extensions, the Custom Layer Extractor and the Custom Layer Operation to both  the Model Optimizer and the Inference Engine. 

Some of the potential reasons for handling custom layers are: some layers in a neural network model might  not be in the openVINO Model Optimizer supported layers list. They are layers that are not natively supported by the openvino Inference engine. These layers can be added to the Inference Engine as custom layers. The custom layer can therefore be defined as any model layer that is not natively supported by the model framework. 

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were run test using the model before and after conversion

The difference between model accuracy pre- and post-conversion was: I did not notice any noticeable loss of detection precision after conversion to IR representation

The size of the model pre- and post-conversion was: there is significant reduction in model size. For example: the faster_rcnn_inception_v2_coco_2018_01_28 was reduced from the original ~170MB total size to ~55MB after conversion.

The inference time of the model pre- and post-conversion was roughly comparable tested on local setup.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are: Retail store monitoring. Building capacity monitoring, such as in restaurant, concert, meetings. Home security detection. 

Each of these use cases would be useful because the app would be able to detect human presence and how many people have entered/exited the premise. This is even more relevant in the COVID world where social distancing is very important.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:


Lighting: It would have huge impact on detection results. Good lighting is important for proper detection and poor lighting would make model hard or even unable to detect humans.

Model accuracy: naturally higher the accuracy the better the detection rate. However, as with lighting, model accuracy is only based on training materials and the inference results can be affected by other factors as well.

Camera focal/length: these would determine whether the video delivered to model for inferencing is good to begin with. Without proper original images even the best model can do do too much.


Conclusion:

For this project I use udacity workspace. The OpenVino version I used is 2019 R3. So some of the OpenVino updates mentioned in the class apply here. The model I used is the public Tensorflow model faster_rcnn_inception_v2_coco which is downloaded using OpenVino model downloader. After converting to IR form I was able to use it for the app and it appears to work well. The app correctly detects the human in video and sent over to the browser. 

The results are stored in the "out" directory with a screenshot and a video file.

One thing I did observe  is that the MTTQ server subscription to total people count in UI doesn't behave as expected. It only increase when duration statistics is submitted. Consequently to display the correct total number and duration, I submit duration when a person is confirmed to exit the frame. Given the time it takes, the total number of people and average duration lags a little after the person exits. But is done so that I can work around the UI limitation that the average duration and total people count cannot be updated separately.  
