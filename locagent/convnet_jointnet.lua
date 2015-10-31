require 'nn'
require 'AlexNet'
require 'QNet'

return function(args)

    local add_history = args.add_history or false
    local prototxt = args.prototxt or '/home/jccaicedoru/bvlc_alexnet/deploy.prototxt'
    local binary = args.binary or '/home/jccaicedoru/bvlc_alexnet/bvlc_alexnet.caffemodel'

    local alexnet = load_alexnet(prototxt, binary)
    local qnet = load_qnet()

    -- Remove soft max layer
    alexnet:remove(alexnet:size())

    -- Remove output layer
    alexnet:remove(alexnet:size())

    -- Remove fc7 layer
    alexnet:remove(alexnet:size())
    alexnet:remove(alexnet:size())
    alexnet:remove(alexnet:size())

    -- Add layer 1
    -- Linear
    if add_history then
      local full_feature_model = nn.Concat(1)
      local num_features = qnet:get(1).weight:size()[2]
      local actions_history = nn.Linear(num_features, 100)
      full_feature_model:add(actions_history)
      full_feature_model:add(qnet:get(1))
      alexnet:add(full_feature_model)
    else
      alexnet:add(qnet:get(1))
    end
    -- ReLU
    alexnet:add(qnet:get(2))

    -- Add layer 2
    -- Linear
    alexnet:add(qnet:get(3))
    -- ReLU
    alexnet:add(qnet:get(4))

    -- Add Q-Values layer
    alexnet:add(qnet:get(5))

    return alexnet
end
