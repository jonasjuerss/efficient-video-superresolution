import os
import time
import argparse
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from dataloader import frames_from_video_file


parser = argparse.ArgumentParser(description='Efficient video superresolution')
parser.add_argument('--saved_model_dir', type=str, default='/models/Desktop/2023-01-16_03-42-54/2120', 
                    help='Checkpoint directory.')
parser.add_argument('--tflite_model_path', type=str, default='models/model.tflite', 
                    help='Path to TFLite model.')
parser.add_argument('--convert_model', type=bool, default=False, 
                    help='Whether to convert the saved model to TFLite.')
parser.add_argument('--quantize_model', type=bool, default=False, 
                    help='Whether to quantize the model during TFLite conversion.')
parser.add_argument('--run_inference', type=bool, default=True, 
                    help='Whether to run inference.')
parser.add_argument('--ground_truth_val_dir', type=str, default='data/val/',
                    help='Directory with ground validation training data.')
parser.add_argument('--tecogan_generated_val_dir', type=str, default='data/val/',
                    help='Directory with HR validation images generated by TecoGAN.')
parser.add_argument('--sequence_length', type=int, default=10,
                    help='The sequence length.')
parser.add_argument('--lr_height', type=int, default=32,
                    help='Height of LR image.')
parser.add_argument('--lr_width', type=int, default=32,
                    help='Width of LR image.')
parser.add_argument('--hr_height', type=int, default=128,
                    help='Height of HR image.')
parser.add_argument('--hr_width', type=int, default=128,
                    help='Width of HR image.')
parser.add_argument('--num_predictions', type=int, default=100,
                    help='Number of predictions to measure inference time.')


if __name__ == "__main__":
    args = parser.parse_args()

    if args.convert_model:
        # Convert model
        converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

        if args.quantize_model:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        # Save the model.
        with open(args.tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        

    if args.run_inference:
        interpreter = tf.lite.Interpreter(args.tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        val_frames = frames_from_video_file(os.path.join(args.ground_truth_val_dir, 'scene_2003'), 
                                        os.path.join(args.tecogan_generated_val_dir, 'scene_2003'),
                                        args.sequence_length,
                                        (args.lr_height, args.lr_width),
                                        x1 = 45, x2 = 60, start = 6)
        input_image, ground_truth, tecogan_generated = val_frames

        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_image, axis = 0))

        starttime = time.time()
        for _ in range(args.num_predictions):
            interpreter.invoke()
        endtime = time.time()
        inference_time = (endtime - starttime) / args.num_predictions

        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_image = output_data[0]

        print('Inference summary:')
        print('Quantization applied? ', args.quantize_model)
        print('Inference time per sequence (in s): ', inference_time)
        print('MAE: ', np.mean(np.abs(output_image - ground_truth)))
        print('MSE: ', np.mean(np.square(output_image - ground_truth)))
        print('SSIM: ', tf.reduce_mean(tf.image.ssim(output_image, ground_truth, max_val = 1)).numpy())
        print('PSNR: ', tf.reduce_mean(tf.image.psnr(output_image, ground_truth, max_val = 1)).numpy())