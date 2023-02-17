from typing import List, Dict
import tensorflow as tf


class TaskModel:
    @classmethod
    def build_model_from_config(cls, config: Dict) -> tf.keras.Model:
        return tf.keras.Model.from_config(config)
