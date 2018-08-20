import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
import pandas as pd
from random import *
from termcolor import colored
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Fork vladimir-paliy/chatbot-retrieval
# Load your own data here
## Insertar respuestas por código
#INPUT_CONTEXT_TERMINAL = "Is it you?"
#POTENTIAL_RESPONSES_TERMINAL = ["Yes", "No", 'I think yes', 'Of course', '17', 'you', 'cp /etc/skel /home/usernam -r __eou__ not the document in it . __eou__ i have no clue . __eou__']

# Fork Shangeli/chatbot-retrieval
# Load your own data here
test_df = pd.read_csv("./data/test.csv")
elementId = randint(0, 10000)
INPUT_CONTEXT = test_df.Context[elementId]
POTENTIAL_RESPONSES = test_df.iloc[elementId,1:].values

def get_features(context, utterances):
  context_matrix = np.array(list(vp.transform([context])))
  #utterance_matrix = np.array(list(vp.transform([utterance])))
  utterance_matrix = np.array(list(vp.transform([utterances[0]])))
  context_len = len(context.split(" "))
  #utterance_len = len(utterance.split(" "))
  utterance_len = len(utterances[0].split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
    "len":len(utterances)
  }

  for i in range(1,len(utterances)):
      utterance = utterances[i];

      utterance_matrix = np.array(list(vp.transform([utterance])))
      utterance_len = len(utterance.split(" "))

      features["utterance_{}".format(i)] = tf.convert_to_tensor(utterance_matrix, dtype=tf.int64)
      features["utterance_{}_len".format(i)] = tf.constant(utterance_len, shape=[1,1], dtype=tf.int64)

  return features, None

if __name__ == "__main__":
  #tf.logging.set_verbosity(tf.logging.INFO)
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  #estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  answerresults=0

## First Example
  print('\n')
  print(colored('First Example:', on_color='on_magenta',color="white"))
  print('\n')

  starttime = time.time()

  if float(tf.__version__[0:4])<0.12: #check TF version to select method
      prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES),as_iterable=True)
  else:
      prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES))

  endtime = time.time()

  if float(tf.__version__[0:4])<0.12: #check TF version to select method
      for r in POTENTIAL_RESPONSES:
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r),as_iterable=True)
          results = next(prob)
          if results>answerresults:
              answerresults=results
  else:
      for r in POTENTIAL_RESPONSES:
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
          results = next(prob)
          if results>answerresults:
              answerresults=results

  print(colored('[     Context]', on_color='on_blue',color="white"),"{}".format(INPUT_CONTEXT))
  print('\n')

  if float(tf.__version__[0:4])<0.12: #check TF version to select method
      for r in POTENTIAL_RESPONSES:
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r),as_iterable=True)
          results = next(prob)
          if answerresults==results:
              print(colored('[      Answer]', on_color='on_green'), "{}: {:g}".format(r, next(prob)[0]))
          else:
              print(colored('[      Answer]', on_color='on_red'), "{}: {:g}".format(r, next(prob)[0]))
  else:
      for r in POTENTIAL_RESPONSES:
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
          results = next(prob)
          if answerresults==results:
              print(colored('[      Answer]', on_color='on_green'), "{}: {:g}".format(r, next(prob)[0]))
          else:
              print(colored('[      Answer]', on_color='on_red'), "{}: {:g}".format(r, next(prob)[0]))

  print('\n')
  print(colored('[Right answer]', on_color='on_green'), POTENTIAL_RESPONSES[0])
  print('\n')
  print(colored('[Predict time]', on_color='on_yellow',color="white"),"%f sec" % round(endtime - starttime,2))
  print('\n')

## Second Example
  print('\n')
  print(colored('Second Example:', on_color='on_magenta',color="white"))
  print('\n')

  INPUT_CONTEXT_TERMINAL = input("Write a context: ")
  print(colored('[     Context]', on_color='on_blue',color="white"),"{}".format(INPUT_CONTEXT_TERMINAL))
  print('\n')

  if float(tf.__version__[0:4])<0.12: #check TF version to select method
      while True:
          pass
          POTENTIAL_RESPONSES_TERMINAL = input("Write an answer: ")
          starttime = time.time()
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT_TERMINAL, POTENTIAL_RESPONSES_TERMINAL),as_iterable=True)
          endtime = time.time()
          print(colored('[  Prediction]', on_color='on_green'), "{}: {:g}".format(POTENTIAL_RESPONSES_TERMINAL, next(prob)[0]))
          print(colored('[Predict time]', on_color='on_yellow',color="white"),"%f sec" % round(endtime - starttime,2))
          print('\n')
  else:
      while True:
          pass
          POTENTIAL_RESPONSES_TERMINAL = input("Write an answer: ")
          starttime = time.time()
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT_TERMINAL, POTENTIAL_RESPONSES_TERMINAL))
          endtime = time.time()
          print(colored('[  Prediction]', on_color='on_green'), "{}: {:g}".format(POTENTIAL_RESPONSES_TERMINAL, next(prob)[0]))
          print(colored('[Predict time]', on_color='on_yellow',color="white"),"%f sec" % round(endtime - starttime,2))
          print('\n')
