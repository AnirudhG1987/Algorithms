from imageai.Classification import ImageClassification
import os
execution_path = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
print(execution_path + "/"+"resnet50_imagenet_tf.2.0-2.h5")
prediction.setModelPath(execution_path + "/"+"resnet50_imagenet_tf.2.0-2.h5")
prediction.loadModel()


predictions, percentage_probabilities = prediction.classifyImage(os.getcwd()+"/images/police.jpg", result_count=5)
for index in range(len(predictions)):
  print(predictions[index] , " : " , percentage_probabilities[index])