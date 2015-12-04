require 'loadcaffe'

function load_alexnet(caffe_prototxt, caffe_binary)
  return loadcaffe.load(caffe_prototxt, caffe_binary, 'cudnn')
end
