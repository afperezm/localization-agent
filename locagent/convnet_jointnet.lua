require 'nn'
require 'AlexNet'
require 'QNet'

return function(args)

    local add_history = args.add_history or true
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

    if add_history then
      local mlp = nn.Concat(1)
      local input_size = qnet:get(1).weight:size()[2]
      mlp:add(nn.Linear(input_size, 100))
      mlp:add(qnet:get(1))
    else
      alexnet:add(qnet:get(1))
    end

    alexnet:add(qnet:get(2))

    alexnet:add(qnet:get(3))

    alexnet:add(qnet:get(4))

    return alexnet
end
