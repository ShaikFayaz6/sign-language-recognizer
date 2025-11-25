import os
import sys
import tensorflow as tf

# Reduce TF log verbosity and ensure TF1 graph mode works under TF2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

image_path = sys.argv[1]

# Read the image data (expects JPEG bytes for the retrained graph input)
with tf.io.gfile.GFile(image_path, 'rb') as f:
    image_data = f.read()

# Load labels
with tf.io.gfile.GFile("logs/output_labels.txt", 'r') as f:
    label_lines = [line.rstrip() for line in f]

# Load frozen graph
with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Run session
with tf.compat.v1.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print(f"{human_string} (score = {score:.5f})")
