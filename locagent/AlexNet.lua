-- AlexNet taken from https://raw.githubusercontent.com/eladhoffer/ImageNet-Training/master/Models/AlexNet.lua

require 'nn'

local features = nn.Sequential()
features:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
features:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
features:add(nn.ReLU(true))
features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
features:add(nn.ReLU(true))
features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, 1000))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()

model:add(features):add(classifier)

return {
  model = model,
  regime = {
    epoch        = {1,    19,   30,   44,   53  },
    learningRate = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4},
    weightDecay  = {5e-4, 5e-4, 0,    0,    0   }
  }
}
