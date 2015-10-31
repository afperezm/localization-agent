-- Q-Network

require 'nn'

function load_qnet()

  local model = nn.Sequential()

  -- fc1
  --model:add(nn.Linear(4196, 1024))
  model:add(nn.Linear(4096, 1024))
  model.modules[#model.modules].weight:normal(0, 0.005)
  model.modules[#model.modules].bias:fill(0.1)

  -- relu1
  model:add(nn.ReLU(true))

  -- fc2
  model:add(nn.Linear(1024, 1024))
  model.modules[#model.modules].weight:normal(0, 0.005)
  model.modules[#model.modules].bias:fill(0.1)

  -- relu2
  model:add(nn.ReLU(true))

  -- q-values
  model:add(nn.Linear(1024, 10))
  model.modules[#model.modules].weight:normal(0, 0.01)
  model.modules[#model.modules].bias:fill(0)

  return model

end
