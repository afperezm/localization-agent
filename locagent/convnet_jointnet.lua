require 'nn'
require 'AlexNet'
require 'QNet'

return function(args)

    local args = args.network_params

    local prototxt = args.prototxt
    local binary = args.binary
    local num_actions = args.num_actions

    local alexnet = load_alexnet(prototxt, binary)
    local qnet = load_qnet(num_actions)

    -- Add layer 1
    -- Linear
    alexnet:add(qnet:get(1))

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
