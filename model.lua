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
    net:add(nn.View(sizeData))
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
    local firstOutputConv = 10
    local firstMaxPool = 1
    local secondInputConv = 5
    local secondOutputConv = 1

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalConvolution(10, secondOutputConv, secondInputConv))
    net:add(nn.ReLU())
    local num = (math.floor((sizeData - firstInputConv + 1) / firstMaxPool) - secondInputConv + 1) * secondOutputConv
    net:add(nn.View(num))
    net:add(nn.Linear(num, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-conv(' ..
        firstOutputConv .. ',' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-' .. num .. '-200-500'

    print('Neural Network 1\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet1()
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

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-' .. num .. '-120-84'

    print('Lenet1\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet2()
    local firstInputConv = 5
    local secondInputConv = 3
    local firstOutputConv = 20
    local secondOutputConv = 10
    local firstMaxPool = 4
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.View(math.floor((sizeData - firstInputConv + 1)/firstMaxPool)*firstOutputConv, 1))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.Linear(num, 120))
    net:add(nn.ReLU())
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    net:add(nn.Linear(84, nb_class))
    -- net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-' .. num .. '-120-84'

    print('Lenet2\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet3()
    local firstInputConv = 5
    local secondInputConv = 3
    local firstOutputConv = 20
    local secondOutputConv = 10
    local firstMaxPool = 4
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.View(math.floor((sizeData - firstInputConv + 1)/firstMaxPool)*firstOutputConv, 1))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.Linear(num, 1024))
    net:add(nn.ReLU())
    net:add(nn.Linear(1024, 512))
    net:add(nn.ReLU())
    net:add(nn.Linear(512, nb_class))
    -- net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-' .. num .. '-120-84'

    print('Lenet2\n' .. net:__tostring())       -- print the model
end


function neuralNetwork_MCDCNN()
    local firstInputConv = 5
    local secondInputConv = 5
    local firstOutputConv = 8
    local secondOutputConv = 4
    local firstMaxPool = 2
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv))
    net:add(nn.Sigmoid())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.View(math.floor((sizeData - firstInputConv + 1)/firstMaxPool)*firstOutputConv, 1))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv))
    net:add(nn.Sigmoid())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.Linear(num, 732))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(732, nb_class))
    -- net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-' .. num .. '-732'

    print('MC-DCNN\n' .. net:__tostring())       -- print the model
end
