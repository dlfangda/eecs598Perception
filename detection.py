
from glob import glob
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import csv
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
def main():
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    testDir = ''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('../betterM' + '/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    testFiles = glob('../../test/*/*_image.jpg')
    counter0=0
    counter1=0
    counter2=0
    print('Opening the csv file')
    f = open('detection.csv',mode = 'w')
    rwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rwriter.writerow(['guid/image', 'u', 'v'])
    print('Reading csv files')
    for file in testFiles:
        image = Image.open(file)
        (im_width, im_height) = image.size

        outputDict = run_inference_for_single_image(image, detection_graph)
        
        for i in range(outputDict['num_detections']):
            if outputDict['detection_classes'][i] >= 2 and outputDict['detection_classes'][i] <= 9:
                tb = int(0.5*(outputDict['detection_boxes'][i][0]+outputDict['detection_boxes'][i][2])*im_height)
                lr = int(0.5*(outputDict['detection_boxes'][i][1]+outputDict['detection_boxes'][i][3])*im_width)
                rwriter.writerow([file[11:-10], lr, tb])
                break
                #bicycle
            
        '''
        result = 0
        counts = [0.0,0.0,0.0]
        changing = False
        for i in range(outputDict['num_detections']):
            if outputDict['detection_classes'][i] == 2:
                #bicycle
                counts[2] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 3:
                #car
                counts[1] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 4:
                #motorcycle
                counts[2] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 5:
                #plane
                counts[0] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 6:
                #bus
                counts[2] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 7:
                #train
                counts[0] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 8:
                #truck
                counts[2] += outputDict['detection_scores'][i]
                changing = True
            elif outputDict['detection_classes'][i] == 9:
                #boat
                counts[0] += outputDict['detection_scores'][i]
                changing = True
            if changing:
                result = np.argmax(counts)
            else:
                result = 0
        if result == 0: counter0 += 1
        if result == 1: counter1 += 1
        if result == 2: counter2 += 1
        '''
        #rwriter.writerow([file[11:-10], str(result)])
    print('Number of 0:',counter0)
    print('Number of 1:',counter1)
    print('Number of 2:',counter2)







def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensorDict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensorDict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensorDict:
            # The following processing is only for single image
                detection_boxes = tf.squeeze(tensorDict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensorDict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensorDict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensorDict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
            outputDict = sess.run(tensorDict,
                                     feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
            outputDict['num_detections'] = int(outputDict['num_detections'][0])
            outputDict['detection_classes'] = outputDict[
                    'detection_classes'][0].astype(np.uint8)
            outputDict['detection_boxes'] = outputDict['detection_boxes'][0]
            outputDict['detection_scores'] = outputDict['detection_scores'][0]
            if 'detection_masks' in outputDict:
                outputDict['detection_masks'] = outputDict['detection_masks'][0]
    return outputDict



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

if __name__ == "__main__":
    main()




