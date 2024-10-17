import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def apply_transformations(image, rotation_range=(-3, 3), zoom_range=(0.9, 1.1), noise_stddev=0.05):
    # Convert degrees to radians for rotation
    radians = tf.random.uniform((), minval=np.radians(rotation_range[0]), maxval=np.radians(rotation_range[1]))

    # Apply rotation
    image_rotated = tfa.image.rotate(image, radians)

    # Apply zoom
    zoom = tf.random.uniform((), minval=zoom_range[0], maxval=zoom_range[1])
    image_zoomed = tf.image.resize(image_rotated, [int(image.shape[0] * zoom), int(image.shape[1] * zoom)])
    image_zoomed = tf.image.resize_with_crop_or_pad(image_zoomed, image.shape[0],
                                                    image.shape[1])  # Crop/pad to original size

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image_zoomed), mean=0.0, stddev=noise_stddev, dtype=tf.float32)
    image_noisy = image_zoomed + noise
    image_noisy = tf.clip_by_value(image_noisy, 0.0, 255.0)  # Clip to ensure valid image pixel values

    return image_noisy
