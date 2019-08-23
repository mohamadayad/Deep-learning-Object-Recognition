import numpy as np
import caffe

net = caffe.Net('./gnetmodels/referencecaffenet/deploy.prototxt', './gnetmodels/referencecaffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('./gnetmodels/ilsvrc_2012_mean.npy').mean(1).mean(1))

transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1,3,227,227)

img = caffe.io.load_image('./gnetmodels/car_queue1.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', img)

output = net.forward()

output['prob'].argmax()

labels_mapping = np.loadtxt('./gnetmodels/synset_words.txt', str, delimiter='\t')
best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels_mapping[best_n]


img = caffe.io.load_image('./gnetmodels/one_car.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', img)

output = net.forward()

output['prob'].argmax()

labels_mapping = np.loadtxt('./gnetmodels/synset_words.txt', str, delimiter='\t')
best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels_mapping[best_n]
