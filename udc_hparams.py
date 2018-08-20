import tensorflow as tf
from collections import namedtuple

# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  91620, #default = 91620
  "The size of the vocabulary. Only change this if you changed the preprocessing")
### Se cogen todas las palabras del udc, se buscan las repetidas y se crea una lista en orden descente
### Este parametro dice el numero de entradas que cogemos del vector donde se guardan cada palabra repetida

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
### Tamaño del vector del Word Embedding, número de relaciones que hace de una palabra ## default = 100

tf.flags.DEFINE_integer("rnn_dim", 128, "Dimensionality of the RNN cell")
### Número de neuronas (probar con 128, 256, 512 y 1056) ## default = 256

tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
### Número máximo de tokens en un contexto ## default = 160

tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")
### Número máximo de tokens en una respuesta ## default = 80

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors") #Glove es una librería para hacer embedding
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
 #alpha #default = 0.001

# batch_size, es el número de ejemplos usados en cada paso del descenso del gradiente
tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training")
### Tamaño del batch (probar con 1, 64, 128 y 256) ## default = 128
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
## default = 16


tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "rnn_dim",
    "vocab_size",
    "glove_path",
    "vocab_path"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim)
