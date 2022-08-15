import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

##v1版本中tensorflow中contrib模块十分丰富，但是发展不可控，
# 因此在v2版本中将这个模块集成到其他模块中去了
#tf=tf.compat.v1##tensorflow的兼容处理

#训练：python main.py --dataset market --train --input_height 128 --output_height 128
#需要指明--train！！！默认是test模式！


####测试：
##python main.py --dataset market --options 5  --output_path gen_market  --sample_size 12000
##向gen_market输出12000张图片
###python resizeImage.py，将生成图片调整为合适的大小
#基本框架，调整对应参数

#tensorflow1.15+cuda10+cudnn7.6.5(不是推荐版本，但是会减少报错)

###目前可以跑了2022/6/7

#If you want to see a list of allocated tensors when OOM happens,
# add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
###显存分配不足，减小batch_size或换用更强大的显卡。


###

flags = tf.app.flags
#flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")###训练轮次，默认25次
flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
###固定学习率 0.0002，问题是收敛速度慢
###迭代下降，或者直接设置小的学习率，让loss更小，但是收敛速度也会变慢
##学习率是控制模型学习的速度，也就是它控制权重更新以达到损失值最小点的速度。如果设置过大，
# 在训练一段时间会出现梯度爆照，通俗点会发现训练误差越来越大，没有拟合趋势。如果过小也有梯度消失的可能。

#flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
###，但是在训练的过程中又出现了G和Dloss不按照正常的情况下降和上升：

#0.001，Epoch: [ 0] [   0/ 202] time: 23.6245, d_loss: 1.39066577, g_loss: 0.61362630

##0.002，判别器太强了，压制了生成器。
##Epoch: [ 0] [   8/ 202] time: 88.2045, d_loss: 7.22423220, g_loss: 0.02918075

##提升G的学习率，降低D的学习率。G训练多次，D训练一次。使用一些更先进的GAN训练目标函数。

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")

##batchsize调整
#flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [64]")
###flags.DEFINE_integer("batch_size", 36, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 128, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("output_path", 'duke_result',"output image path")
flags.DEFINE_integer("sample_size", 1000,"How much sample you want to output")
flags.DEFINE_integer("options", 1,"output option")#option1，输出指令
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
#输入类型为.jpg
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")

##训练过程需要手动输入，默认test模式
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.compat.v1.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.compat.v1.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    OPTION = FLAGS.options
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.compat.v1.app.run()
