import tensorflow as tf
from keras.callbacks import TensorBoard


class BatchedTensorBoard(TensorBoard):
    def __init__(self):
        self.log_dir        = "./logs"
        self.batch_writer   = tf.summary.FileWriter(self.log_dir)
        self.step           = 0
        super().__init__(self.log_dir)

        def on_batch_end(self, batch, logs={}):
            for name, value in logs.items():
                if name in ["batch", "size"]:
                    continue
                summary                     = tf.Summary()
                summary_value               = summary.value.add()
                summary_value.simple_value  = value.item()
                summary_value.tag           = name
                self.writer.add_summary(summary, self.step)
            self.writer.flush()
            self.step += 1
