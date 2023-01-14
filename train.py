import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import wandb

from dataloader import FrameGenerator, frames_from_video_file
from wandb_utils import init_wandb, log
from custom_rnn import CustomRNN
from losses import Losses

parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--seed', type=int, default=1,
                    help='The seed.')
parser.set_defaults(use_wandb=True)
parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                    help='Turns off logging to wandb')
parser.add_argument('--sequence_length', type=int, default=10,
                    help='The sequence length.')
parser.add_argument('--num_res_blocks', type=int, default=16,
                    help='Number of residual blocks.')
parser.add_argument('--num_filters', type=int, default=64,
                    help='The sequence length.')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='Kernel size.')
parser.add_argument('--strides', type=int, default=1,
                    help='Strides.')
parser.add_argument('--padding', type=str, default='same',
                    help='Padding.')
parser.add_argument('--block_size', type=int, default=4,
                    help='Block size for upsampling.')
parser.add_argument('--lr_height', type=int, default=32,
                    help='Height of LR image.')
parser.add_argument('--lr_width', type=int, default=32,
                    help='Width of LR image.')
parser.add_argument('--hr_height', type=int, default=128,
                    help='Height of HR image.')
parser.add_argument('--hr_width', type=int, default=128,
                    help='Width of HR image.')
parser.add_argument('--ground_truth_train_dir', type=str, default='./data/train/',
                    help='Directory with ground truth training data.')
parser.add_argument('--ground_truth_val_dir', type=str, default='./data/val/',
                    help='Directory with ground truth validation data.')
parser.add_argument('--tecogan_generated_train_dir', type=str, default='./data/train/',
                    help='Directory with HR training images generated by TecoGAN.')
parser.add_argument('--tecogan_generated_val_dir', type=str, default='./data/val/',
                    help='Directory with HR validation images generated by TecoGAN.')
parser.add_argument('--alpha', type=float, default=1,
                    help='Weighting parameter for distillation loss.') # TODO Check default value for alpha
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Epochs.')
parser.add_argument('--initial_learning_rate', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--decay_steps', type=int, default=50000,
                    help='Number of steps for exponential learning rate decay.')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='Rate of exponential learning rate decay.')
parser.add_argument('--checkpoint_freqs', type=int, default=5,
                    help='Epoch frequency for model checkpointing.')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                    help='Directory to save model checkpoints.')


if __name__ == "__main__":
    args = parser.parse_args()
    args = init_wandb(args)

    custom_model = CustomRNN(args)
    model = custom_model.get_model()
    
    
    output_signature = (tf.TensorSpec(shape=(args.sequence_length, args.lr_height, args.lr_width, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(args.sequence_length, args.hr_height, args.hr_width, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(args.sequence_length, args.hr_height, args.hr_width, 3), dtype=tf.float32))

    train_ds = tf.data.Dataset.from_generator(FrameGenerator(args.ground_truth_train_dir, args.tecogan_generated_train_dir, args.sequence_length, output_size = (args.lr_height, args.lr_width), training=True),
                                              output_signature=output_signature)
    train_ds_size = len(list(train_ds))
    val_ds = tf.data.Dataset.from_generator(FrameGenerator(args.ground_truth_val_dir, args.tecogan_generated_val_dir, args.sequence_length, output_size = (args.lr_height, args.lr_width), training=False),
                                            output_signature=output_signature)
    val_ds_size = len(list(val_ds))
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE).batch(args.batch_size)
    val_ds = val_ds.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE).batch(args.batch_size)

    val_frames = frames_from_video_file(os.path.join(args.ground_truth_train_dir, 'scene_2007'), 
                                        os.path.join(args.ground_truth_train_dir, 'scene_2007'),
                                        args.sequence_length,
                                        (args.lr_height, args.lr_width),
                                        x1 = 40, x2 = 25, start = 0)[0][None, ...]

    losses = Losses(args)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.initial_learning_rate, args.decay_steps, args.decay_rate, staircase = False, name = None)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

    for epoch in range(args.epochs):
        # Train
        student_loss_epoch, distillation_loss_epoch = 0, 0
        for x, ground_truth, teacher_output in train_ds:
            loss_values, grads = losses.train_step(model, x, ground_truth, teacher_output)
            student_loss_batch, distillation_loss_batch = loss_values
            student_loss_epoch += student_loss_batch
            distillation_loss_epoch += distillation_loss_batch
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss_epoch = student_loss_epoch + distillation_loss_epoch
        log({'student_loss': student_loss_epoch / train_ds_size, 
             'distillation_loss': distillation_loss_epoch / train_ds_size, 
             'total_loss': total_loss_epoch / train_ds_size}, 
             step = epoch)
        if epoch % args.checkpoint_freqs == 0:
            model.save(args.checkpoint_dir)
    

        # Validation
        student_loss_val, distillation_loss_val, mse, psnr, ssim = 0, 0, 0, 0, 0
        for x, ground_truth, teacher_output in val_ds:
            loss_values, student_output = losses.test_step(model, x, ground_truth, teacher_output)
            student_loss_batch, distillation_loss_batch = loss_values
            student_loss_val += student_loss_batch
            distillation_loss_val += distillation_loss_batch

            mse += tf.keras.losses.MeanSquaredError()(student_output, ground_truth)
            psnr += tf.reduce_mean(tf.image.psnr(student_output, ground_truth, max_val = 1))
            ssim += tf.reduce_mean(tf.image.ssim(student_output, ground_truth, max_val = 1))

        total_loss_val = student_loss_val + distillation_loss_val
        log({'student_loss_val': student_loss_val / val_ds_size, 
             'distillation_loss_val': distillation_loss_val / val_ds_size, 
             'total_loss_val': total_loss_val / val_ds_size,
             'mse_val': mse / val_ds_size,
             'psnr_val': psnr / val_ds_size,
             'ssim_val': ssim / val_ds_size}, 
             step = epoch)

        train_image = np.moveaxis((model(val_frames).numpy() * 255).astype(np.uint8), -1, 2)
        log({'train_image': wandb.Video(train_image)}, step = epoch)        