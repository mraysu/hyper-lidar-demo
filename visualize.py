#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
from auxiliary.dataset import SemKITTI_sk
from torch.utils.data import DataLoader
import torch
import numpy as np

import csv
import datetime

def collate_fn_BEV(data):
    #data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    #label2stack = np.stack([d[1] for d in data]).astype(np.int32)
    #grid_ind_stack = [d[2] for d in data]
    #point_label = [d[3] for d in data]
    #xyz = [d[4] for d in data]
    points = data[0][0]
    remissions = data[0][1]
    labels = data[0][2]
    viridis_colors = data[0][3]
    sem_label_colors = data[0][4]
    return points, remissions, labels, viridis_colors, sem_label_colors

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      default=None,
      required=False,
      help='Alternate location for labels, to use predictions folder. '
      'Must point to directory containing the predictions in the proper format '
      ' (see readme)'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-o',
      dest='do_instances',
      default=False,
      required=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_images', '-r',
      dest='ignore_images',
      default=False,
      required=False,
      action='store_true',
      help='Visualize range image projections too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--link', '-l',
      dest='link',
      default=False,
      required=False,
      action='store_true',
      help='Link viewpoint changes across windows. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      required=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
    '--color_learning_map',
    dest='color_learning_map',
    default=False,
    required=False,
    action='store_true',
    help='Apply learning map to color map: visualize only classes that were trained on',
  )
  parser.add_argument(
    '--debug_auto',
    dest='debug_auto',
    default=False,
    required=False,
    action='store_true'
  )
  parser.add_argument(
    '--shuffle',
    dest='shuffle',
    default=False,
    required=False,
    action='store_true',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("Predictions", FLAGS.predictions)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("do_instances", FLAGS.do_instances)
  print("ignore_images", FLAGS.ignore_images)
  print("link", FLAGS.link)
  print("ignore_safety", FLAGS.ignore_safety)
  print("color_learning_map", FLAGS.color_learning_map)
  print("offset", FLAGS.offset)
  print("debug_auto", FLAGS.debug_auto)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print(f"Sequence folder {scan_paths} exists! Using sequence from {scan_paths}")
  else:
    print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # does sequence folder exist?
  if not FLAGS.ignore_semantics:
    if FLAGS.predictions is not None:
      label_paths = os.path.join(FLAGS.predictions, "sequences",
                                 FLAGS.sequence, "predictions")
    else:
      label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
    if os.path.isdir(label_paths):
      print(f"Labels folder {label_paths} exists! Using labels from {label_paths}")
    else:
      print(f"Labels folder {label_paths} doesn't exist! Exiting...")
      quit()

    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    # check that there are same amount of labels and scans
    if not FLAGS.ignore_safety:
      assert(len(label_names) == len(scan_names))

  color_dict = CFG["color_map"]
  if FLAGS.color_learning_map:
    learning_map_inv = CFG["learning_map_inv"]
    learning_map = CFG["learning_map"]
    color_dict = {key:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

  mydataset = SemKITTI_sk(data_path = "dataset/sequences", 
                          imageset="train", 
                          label_mapping="config/semantic-kitti-00.yaml",
                          sem_color_dict=color_dict,
                          percentLabels=1)
  
  dataloader = DataLoader(mydataset, batch_size=1, shuffle=FLAGS.shuffle, collate_fn=collate_fn_BEV, num_workers=0)

  # create a scan
  #if FLAGS.ignore_semantics:
  #  scan = LaserScan(dataset=mydataset, project=False)  # project all opened scans to spheric proj
  #else:
  #  scan = SemLaserScan(sem_color_dict=color_dict, dataset=mydataset, project=False)

  # create a visualizer
  semantics = not FLAGS.ignore_semantics
  instances = FLAGS.do_instances
  images = not FLAGS.ignore_images
  if not semantics:
    label_names = None
  
  now = datetime.datetime.now()
  with open('timedata/{0}.csv'.format('{0}'.format(now).replace(" ", "_").replace(":", "-")[:19]), 'w', newline='') as csvfile:

    fieldNames = ['Points','LoadData', 'PlotRaw','PlotSem']
    writer = None
    writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
    writer.writeheader()
    vis = LaserScanVis(dataloader=dataloader,
                      dataset=mydataset,
                      scan_names=scan_names,
                      label_names=label_names,
                      offset=FLAGS.offset,
                      csvwriter=writer,
                      debug_auto=FLAGS.debug_auto,
                      semantics=semantics, instances=instances and semantics, images=images, link=FLAGS.link)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()
