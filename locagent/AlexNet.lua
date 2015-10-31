require 'loadcaffe'

function load_alexnet(args)
  return loadcaffe.load(args.prototxt, args.binary)
end
