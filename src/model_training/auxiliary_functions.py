import numpy as np
import tensorflow as tf


def dataset_to_tensors(dataset: dict) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Convert a dataset dict of DatasetEntry objects into TensorFlow tensors.
    """
    images, masks = [], []
    for entry in dataset.values():
        images.append(np.expand_dims(entry.nn_input_image, axis=-1).tolist())
        masks.append(entry.segmentation_mask.tolist())
    return (
        tf.convert_to_tensor(images, dtype=tf.float32),
        tf.convert_to_tensor(masks, dtype=tf.int32),
    )