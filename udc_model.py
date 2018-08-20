import tensorflow as tf
import sys

def get_id_feature(features, key, len_key, max_len):
  ids = features[key]
  ids_len = tf.squeeze(features[len_key], [1])
  ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
  return ids, ids_len

def create_train_op(loss, hparams):
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=hparams.learning_rate,
      clip_gradients=10.0,
      optimizer=hparams.optimizer)
  return train_op


def create_model_fn(hparams, model_impl):

  def model_fn(features, targets, mode):
    context, context_len = get_id_feature(
        features, "context", "context_len", hparams.max_context_len)

    utterance, utterance_len = get_id_feature(
        features, "utterance", "utterance_len", hparams.max_utterance_len)

    #batch_size = targets.get_shape().as_list()[0]

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      probs, loss = model_impl(
          hparams,
          mode,
          context,
          context_len,
          utterance,
          utterance_len,
          targets) # respuesta objetivo
      train_op = create_train_op(loss, hparams)
      return probs, loss, train_op

####################################################################################
####################################################################################

      #Guardar probs para TensorBoard
      #tf.summary.scalar("accuracy", probs)
      #tf.summary.histogram("accuracy", probs)

      #Guardar loss para TensorBoard
      tf.summary.scalar("loss", loss)
      tf.summary.histogram("loss", loss)

      #Guardar train_op para TensorBoard
      tf.summary.histogram("train_op", train_op)

      # Merge all summaries into a single op
      merge = tf.summary.merge_all()

      with tf.Session() as sess:
          #Run the initializer
          sess.run(init)
          #train_writer = tf.summary.FileWriter('./logs/TFG/train/', sess.graph)
          train_writer = tf.summary.FileWriter('./logs/TFG/train/', graph=tf.get_default_graph())
          logs_path, graph=tf.get_default_graph()

      #sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      #uninited = sess.run([tf.report_uninitialized_variables()])
      #print("tf.report_uninitialized_variables() len = {}".format(uninited))

      # Start input enqueue threads.
      #coord = tf.train.Coordinator()
      #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      #step = 0
      #try:
        #while not coord.should_stop() and step < FLAGS.train_steps:
          #results = sess.run([predictions, loss, train_op, merged])
          #step += 1
          #if step % 100:
            #print("step: {:06d} loss: {:04f}".format(step, results[1]))
            #train_writer.add_summary(results[3], step)
      #except tf.errors.OutOfRangeError:
        #print('Done for %d epoch.' % (1))
      #finally:
        # When done, ask the threads to stop.
        #coord.request_stop()

    # Wait for threads to finish.
    #coord.join(threads)
    #sess.close()

      #model = DNN(network, tensorboard_verbose=3)

####################################################################################
####################################################################################

    if mode == tf.contrib.learn.ModeKeys.INFER:
      probs, loss = model_impl(
          hparams,
          mode,
          context,
          context_len,
          utterance,
          utterance_len,
          None) #intenta adivinar la respuesta
      return probs, 0.0, None

    if mode == tf.contrib.learn.ModeKeys.EVAL:
      #Lo he cambiado de arriba a aquí
      batch_size = targets.get_shape().as_list()[0]

      #Guardar batch_size para TensorBoard
      tf.summary.histogram("batch_size", batch_size)

      # We have 10 exampels per record, so we accumulate them
      all_contexts = [context]
      all_context_lens = [context_len]
      all_utterances = [utterance]
      all_utterance_lens = [utterance_len]
      all_targets = [tf.ones([batch_size, 1], dtype=tf.int64)] #le pasamos las respuestas

      for i in range(9):
        distractor, distractor_len = get_id_feature(features,
            "distractor_{}".format(i),
            "distractor_{}_len".format(i),
            hparams.max_utterance_len)
        all_contexts.append(context)
        all_context_lens.append(context_len)
        all_utterances.append(distractor)
        all_utterance_lens.append(distractor_len)
        all_targets.append(
          tf.zeros([batch_size, 1], dtype=tf.int64)
        )

      probs, loss = model_impl(
          hparams,
          mode,
          tf.concat(0, all_contexts),
          tf.concat(0, all_context_lens),
          tf.concat(0, all_utterances),
          tf.concat(0, all_utterance_lens),
          tf.concat(0, all_targets))

      split_probs = tf.split(0, 10, probs)
      shaped_probs = tf.concat(1, split_probs)

      # Add summaries
      tf.histogram_summary("eval_correct_probs_hist", split_probs[0])
      tf.scalar_summary("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))
      tf.histogram_summary("eval_incorrect_probs_hist", split_probs[1])
      tf.scalar_summary("eval_incorrect_probs_average", tf.reduce_mean(split_probs[1]))

      return shaped_probs, loss, None

  return model_fn
