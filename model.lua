require 'nn';
require 'cunn';
require 'paths';

function printTitleModel()
    print('--------------------------------------------------------------')
    print('-------------------- Model Neural Network --------------------')
    print('--------------------------------------------------------------')
end

function neuralNetwork1(net, sizeData, nb_class)
    local net = nn.Sequential()
    net:add(nn.TemporalConvolution(1, 1, 5))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(5))
    net:add(nn.TemporalConvolution(1, 1, 5))
    net:add(nn.ReLU())
    net:add(nn.View(sizeData/5 - 5))
    net:add(nn.Linear(sizeData/5 - 5, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    print('Neural Network 1\n' .. net:__tostring())       -- print the model

    return net
end


function neuralNetwork2(net, sizeData, nb_class)
    local net = nn.Sequential()
    net:add(nn.TemporalConvolution(1, 1, 10))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(10))
    net:add(nn.TemporalConvolution(1, 1, 5))
    net:add(nn.ReLU())
    net:add(nn.View(sizeData/10 - 5))
    net:add(nn.Linear(sizeData/10 - 5, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    print('Neural Network 2\n' .. net:__tostring())       -- print the model

    return net
end


function neuralNetwork3(net, sizeData, nb_class)
    local net = nn.Sequential()
    net:add(nn.TemporalConvolution(1, 1, 10))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(10))
    net:add(nn.TemporalConvolution(1, 1, 10))
    net:add(nn.ReLU())
    net:add(nn.View(sizeData/10 - 10))
    net:add(nn.Linear(sizeData/10 - 10, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    print('Neural Network 3\n' .. net:__tostring())       -- print the model

    return net
end


function neuralNetworkLenet(net, sizeData, nb_class)
    local net = nn.Sequential()
    net = nn.Sequential()
    net:add(nn.TemporalConvolution(1, 1, 5))     -- 3 input image channels, 6 outputs channels, 5x5 convolution kernel
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(2))      -- A max-pooling operation that looks at 2x2 windows and finds the max
    net:add(nn.TemporalConvolution(1, 3, 5))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(2))
    local num = (  math.floor((math.floor((sizeData-5+1)/2) - 5 + 1)/2)*3  )
    net:add(nn.View(num))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.Linear(num, 120))             -- fully connected layer (matrix multiplication between inputs and weights)
    -- net:add(nn.View(96))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    -- net:add(nn.Linear(96, 120)             -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    net:add(nn.Linear(84, nb_class))                  -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.LogSoftMax())                    -- converts the output to a log-probability. Useful for classification problems

    print('Lenet\n' .. net:__tostring())       -- print the model

    return net
end
