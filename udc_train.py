import os
import time
import itertools
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model
import logging
#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
#Logging levels for INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3 respectively.

tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level") #default 20
#tf.logging._level_names    ## outputs => {50: 'FATAL', 40: 'ERROR', 30: 'WARN', 20: 'INFO', 10: 'DEBUG'}
#tf.logging.get_verbosity() ## outputs => 30 (default value)
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.") #Tamaño total udc = una epoch
### Para tener el mismo numero de steps siempre en el corpus y basarme en los mismo datos siempre
### cambio las epochs a 1, default None
tf.flags.DEFINE_integer("eval_every", 251, "Evaluate after this many train steps")
## Evalua una vez lo que llevamos cada cierto número de steps, default 2000

FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

if FLAGS.model_dir:
  MODEL_DIR = FLAGS.model_dir
else:
  MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

#tf.logging.set_verbosity(tf.logging.INFO)
#logging.getLogger("tensorflow").setLevel(logging.WARNING)
tf.logging.set_verbosity(FLAGS.loglevel)
#tf.train.Saver(write_version=tf.train.SaverDef.V2)

start = time.process_time()

def main(unused_argv):
  hparams = udc_hparams.create_hparams()

  model_fn = udc_model.create_model_fn(
    hparams,
    model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig())

  input_fn_train = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    input_files=[TRAIN_FILE],
    batch_size=hparams.batch_size,
    num_epochs=FLAGS.num_epochs)

  input_fn_eval = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size,
    num_epochs=1)

  eval_metrics = udc_metrics.create_evaluation_metrics()
  
  eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
	eval_steps=1,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

  estimator.fit(input_fn=input_fn_train, steps=20201, monitors=[eval_monitor])
## steps es el número máximo de steps que se entrenarán, default None

if __name__ == "__main__":
  tf.app.run()
