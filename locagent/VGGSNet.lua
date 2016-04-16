require 'loadcaffe'

function load_vggsnet(caffe_prototxt, caffe_binary)

  -- Load model
  local vggsnet = loadcaffe.load(caffe_prototxt, caffe_binary, 'cudnn')

  -- Remove soft max layer
  vggsnet:remove(vggsnet:size())

  -- Remove output layer
  vggsnet:remove(vggsnet:size())

  -- Remove fc7 layer
  vggsnet:remove(vgssnet:size())
  vggsnet:remove(vggsnet:size())
  vggsnet:remove(vggsnet:size())

  return vggsnet
end
