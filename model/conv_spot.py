from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle as pkl

import img_slicer, sys, cv2, find_boxes, os

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # sunspot slices are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def train_and_eval():
  # Load training and eval data
  train_dict = img_slicer.build_img_set("./train")
  train_data = np.asarray(train_dict["slices"],dtype=np.float16)
  train_labels = np.array(train_dict["labels"])
  eval_dict = img_slicer.build_img_set("./test")
  eval_data = np.asarray(eval_dict["slices"],dtype=np.float16)
  eval_labels = np.array(eval_dict["labels"])

  # Create the Estimator
  sunspot_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./model/ckpt")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  sunspot_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = sunspot_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

def predict():
    skip_ratio = 1 # get all the slices
    stride = 12 # lower for more accurany, higher for faster prediction
    filter_size = 28 # always the same for this architecture
    dir = "./test"
    ext = ".jpg"
    filenames = [os.path.join(dir,fn) for fn in os.listdir(dir) if ext in fn]
    #filenames = ["./test/venustransit_image_can_visible_thumb_latest.jpg"]

    # Create the Estimator
    sunspot_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./model/ckpt")

    for fn in filenames:
        print("finding sunspots in:",fn)
        # Load training and eval data
        img = cv2.imread(fn,0)
        slices,_ = img_slicer.slice_and_label(img,[],
            skip_ratio=skip_ratio,stride=stride)
        pred_data = np.asarray(slices,dtype=np.float16)



        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Evaluate the model and print results
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": pred_data},
            num_epochs=1,
            batch_size=1000,
            shuffle=False)
        pred_results = sunspot_classifier.predict(input_fn=pred_input_fn)
        pred_classes, pred_probab = [],[]
        for p in pred_results:
            pred_classes.append(p["classes"])
            pred_probab.append(p["probabilities"])

        boxes = find_boxes.find(classes=pred_classes,
                                probabilities=pred_probab,
                                img_shape=img.shape,
                                filter_size=filter_size,
                                stride=stride)

        for box in boxes:
            cv2.rectangle(img,(box[0],box[2]),(box[1],box[3]),(255,0,0),2)

        cv2.imwrite(fn.rsplit("/")[-1],img)

        #cv2.imshow("sunspots",img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def main(unused_argv):
    if len(sys.argv) < 2:
        print("Usage: python conv_spot.py [mode]")
    elif sys.argv[1] == "train":
        train_and_eval()
    elif sys.argv[1] == "predict":
        predict()
    else:
        print(sys.argv[1],"is not a recognised mode")




if __name__ == "__main__":
  tf.app.run()
