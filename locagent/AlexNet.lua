require 'loadcaffe'

function load_alexnet(caffe_prototxt, caffe_binary)

  -- Load model
  local alexnet = loadcaffe.load(caffe_prototxt, caffe_binary, 'cudnn')

  -- Remove soft max layer
  alexnet:remove(alexnet:size())

  -- Remove output layer
  alexnet:remove(alexnet:size())

  -- Remove fc7 layer
  alexnet:remove(alexnet:size())
  alexnet:remove(alexnet:size())
  alexnet:remove(alexnet:size())

  return alexnet
end
