
""" 
  The script purpose for Multi Human Parsing Dataset(MHP) v2.0
  1. generate semantic segmentation without instance 
  2. convert to tfrecord

  necessary folder structure - MHP v2.0
  + LV-MHP-v2
    + images
    + list
      - train.txt
      - val.txt
    + parsing_annos
      - *.jpg
  + tfrecord

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import build_data
import tensorflow as tf
import numpy as np
import glob
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder', '../../../DATA/LV-MHP-v2/images/', 'Folder containing images.')
tf.app.flags.DEFINE_string('MuliInst_semantic_segmentation_folder', '../../../DATA/LV-MHP-v2/parsing_annos/','Folder containing semantic segmentation annotations.')
tf.app.flags.DEFINE_string('semantic_segmentation_folder', '../../../DATA/LV-MHP-v2/parsing_annos_no_instance/','Folder containing semantic segmentation annotations.')
tf.app.flags.DEFINE_string('list_folder', '../../../DATA/LV-MHP-v2/list/', 'Folder containing lists for training and validation')
tf.app.flags.DEFINE_string('output_dir', '../../../DATA/_tfrecord/LV-MHP-v2_no_inst/', 'Path to save converted SSTable of TensorFlow examples.')

os.makedirs(FLAGS.output_dir, exist_ok=True)

_NUM_SHARDS = 4
_MHP_MAX_ENTRY = 60
_MAX_PHOTO_NO = 26000
_MAX_PERSON_NO_IN_PHOTO = 30

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  cnt = 0
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(FLAGS.output_dir,'%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d - %s' % (i + 1, len(filenames), shard_id, filenames[i]))
        sys.stdout.flush()

        # Read the image.
        image_filename = os.path.join(FLAGS.image_folder, filenames[i] + '.jpg')
        if os.path.isfile(image_filename) == False:
          continue
        cnt += 1
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)

        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(FLAGS.semantic_segmentation_folder, filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()
  print(cnt)


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


def create_pascal_label_colormap(DATASET_MAX_ENTRIES):
  colormap = np.zeros((DATASET_MAX_ENTRIES, 3), dtype=int)
  ind = np.arange(DATASET_MAX_ENTRIES, dtype=int)
  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3
  return colormap


def generate_parsing_annos_no_instance():
  in_folder_path = FLAGS.MuliInst_semantic_segmentation_folder
  out_folder_path = FLAGS.semantic_segmentation_folder
  os.makedirs(out_folder_path, exist_ok=True)
  for i in range(_MAX_PHOTO_NO):
    for j in range(_MAX_PERSON_NO_IN_PHOTO):
      photo_name = os.path.join(in_folder_path, str(i) + '_' + str(j).zfill(2) + '_01.png')
      if os.path.isfile(photo_name):
        print(photo_name)

        img = np.array(Image.open(photo_name))
        if len(img.shape) == 3:
          img = img[:,:,0]
        for k in range(j+1):
          if k == 0: continue
          if k == 1: continue
          sub_photo_name = os.path.join(in_folder_path, str(i) + '_' + str(j).zfill(2) + '_' + str(k).zfill(2)+'.png')
          print(sub_photo_name)
          tmp_img = np.array(Image.open(sub_photo_name))[:,:,0] 
          if len(tmp_img.shape) == 3:
            tmp_img = tmp_img[:,:,0]
          img = np.maximum(img, tmp_img)
        Image.fromarray(img.astype(dtype=np.uint8)).save(os.path.join(out_folder_path, str(i)+'.png'),'png')


def vis_parsing_annos(in_folder_path, out_folder_path):
  file_list = glob.glob(in_folder_path + '\\*.png')
  os.makedirs(out_folder_path, exist_ok=True)
  for i in file_list:
    print(i)
    out_file_path = os.path.join(out_folder_path, os.path.basename(i)[:-4] + '_c.png')
    img = np.array(Image.open(i))
    color_map = create_pascal_label_colormap(_MHP_MAX_ENTRY)
    Image.fromarray(color_map[img].astype(dtype=np.uint8)).save(out_file_path, 'png')


if __name__ == '__main__':
  #generate_parsing_annos_no_instance()
  #vis_parsing_annos(FLAGS.MuliInst_semantic_segmentation_folder, "../../../DATA/LV-MHP-v2/vis_MHP")
  #vis_parsing_annos(FLAGS.semantic_segmentation_folder, "../../../DATA/LV-MHP-v2/vis_MHP_no_inst" )
  tf.app.run()
