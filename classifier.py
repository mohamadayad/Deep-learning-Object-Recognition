import numpy as np
import caffe

class Classifier(caffe.Ne):
    def __init__(self. model_file, pretrained_file, image_dims=None,
                mean=None, input_scale=None, raw_scale=None,
                channel_swap=None):
        caffe.Net.init(self, model_file, pretrained_file, caffe.TEST)

        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, inputs, oversample=True):
        input_ = np.zeros((len(inputs),
                            self.image_dims[0],
                            self.image_dims[1],
                            self.image_dims[2]),
                            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            center = cp.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                    -self.crop_dims / 2.0,
                    self.crop_dims / 2.0])
            crop = crop.astype(int)
            input_ = input_[:,crop[0]:crop[2], crop[1]:crop[3], :]

        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]], dtype=np.float32)

        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})

        predictions = out[self.outputs[0]]

        if oversample:
            predictions = predictions.reshape((len(predictions / 10, 10, -1)))
            predictions = predictions.mean(1)
        return predictions


