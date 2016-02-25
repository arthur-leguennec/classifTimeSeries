------------------------------------------------------------------------------
---------------- Creation file : 02/22/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for define models ------------------------
------------------------------------------------------------------------------

function printTitleModel()
    print('--------------------------------------------------------------')
    print('-------------------- Model Neural Network --------------------')
    print('--------------------------------------------------------------')
end

function neuralNetwork0()
    net:add(nn.View(1, sizeData))
    net:add(nn.ReLU())
    net:add(nn.Linear(sizeData, 200))
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 200))
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    print('Neural Network 0\n' .. net:__tostring())       -- print the model
end


function neuralNetwork1()
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
end


function neuralNetwork2()
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
end


function neuralNetwork3()
    net:add(nn.TemporalConvolution(1, 1, 10))
    net:add(nn.ReLU())
    net:add(nn.Dropout())
    net:add(nn.TemporalMaxPooling(10))
    net:add(nn.TemporalConvolution(1, 1, 10))
    net:add(nn.ReLU())
    net:add(nn.Dropout())
    net:add(nn.View(sizeData/10 - 10))
    net:add(nn.Linear(sizeData/10 - 10, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Dropout())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Dropout())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    print('Neural Network 3\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet()
    firstInputConv = 5
    secondInputConv = 10
    firstOutputConv = 1
    secondOutputConv = 3
    firstMaxPool = 2
    secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.Linear(num, 120))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout())
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout())
    net:add(nn.Linear(84, nb_class))
    net:add(nn.LogSoftMax())

    print('Lenet\n' .. net:__tostring())       -- print the model
end
