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

    id = '200-200-500 (fully connected)'

    print('Neural Network 0\n' .. net:__tostring())       -- print the model
end


function neuralNetwork1()
    local firstInputConv = 5
    local firstOutputConv = 1
    local firstMaxPool = 5
    local secondInputConv = 5
    local secondOutputConv = 1

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv))
    net:add(nn.ReLU())
    local num = (math.floor((sizeData - firstInputConv + 1) / firstMaxPool) * firstOutputConv - secondInputConv + 1) * secondOutputConv
    net:add(nn.View(num))
    net:add(nn.Linear(num, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    id = 'conv(' .. firstInputConv .. ',' .. firstOutputConv .. ')-maxPool(' .. firstMaxPool ..
        ')-conv(' .. secondInputConv .. ',' .. secondOutputConv .. ')-' .. num .. '-200-500'

    print('Neural Network 1\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet()
    local firstInputConv = 5
    local secondInputConv = 3
    local firstOutputConv = 1
    local secondOutputConv = 3
    local firstMaxPool = 2
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout(0.5))
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout(0.5))
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.Linear(num, 120))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout(0.5))
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    -- net:add(nn.Dropout())
    net:add(nn.Linear(84, nb_class))
    -- net:add(nn.LogSoftMax())

    id = 'conv(' .. firstInputConv .. ',' .. firstOutputConv .. ')-maxPool(' .. firstMaxPool ..
        ')-conv(' .. secondInputConv .. ',' .. secondOutputConv .. ')-maxPool(' .. secondMaxPool ..
        ')-' .. num .. '-120-84'

    print('Lenet\n' .. net:__tostring())       -- print the model
end
