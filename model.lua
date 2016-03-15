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
    net:add(nn.Linear(sizeData, 200))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(200, 1024))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(1024, 2048))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(2048, 2048))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(2048, 1024))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(1024, 512))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(512, 128))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(128, 256))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(256, 1024))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(1024, 512))        -- fully connected layer (matrix multiplication between inputs and weights)
    net:add(nn.Sigmoid())
    net:add(nn.Linear(512, nb_class))

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
    -- net:add(nn.ReLU())
    net:add(nn.TemporalConvolution(10, secondOutputConv, secondInputConv, 1))
    -- net:add(nn.ReLU())
    local num = (math.floor((sizeData - firstInputConv + 1) / firstMaxPool) - secondInputConv + 1) * secondOutputConv
    net:add(nn.View(num))
    net:add(nn.Linear(num, 200))        -- fully connected layer (matrix multiplication between inputs and weights)
    -- net:add(nn.ReLU())
    net:add(nn.Linear(200, 500))        -- fully connected layer (matrix multiplication between inputs and weights)
    -- net:add(nn.ReLU())
    net:add(nn.Linear(500, nb_class))
    net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-conv(' ..
        firstOutputConv .. ',' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-' .. num .. '-200-500'

    print('Neural Network 1\n' .. net:__tostring())       -- print the model
end


-- Model Lenet
-- This model

function neuralNetworkLenet1()
    local firstInputConv = 5
    local secondInputConv = 3
    local firstOutputConv = 1
    local secondOutputConv = 3
    local firstMaxPool = 2
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.Linear(num, 120))
    net:add(nn.ReLU())
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    net:add(nn.Linear(84, nb_class))

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-' .. num .. '-120-84'

    print('Lenet1\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet2()
    local firstInputConv = 5
    local secondInputConv = 3
    local firstOutputConv = 10
    local secondOutputConv = 3
    local firstMaxPool = 4
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.View(math.floor((sizeData - firstInputConv + 1)/firstMaxPool)*firstOutputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    net:add(nn.View(num))
    net:add(nn.ReLU())
    net:add(nn.Linear(num, num/3))
    net:add(nn.ReLU())
    net:add(nn.Linear(num/3, num/6))
    net:add(nn.ReLU())
    net:add(nn.Linear(num/6, nb_class))
    -- net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-' .. num .. '-'.. num/3 .. '-' .. num/6

    print('Lenet2\n' .. net:__tostring())       -- print the model
end


function neuralNetworkLenet3()
    local firstInputConv = 5
    local secondInputConv = 3
    local firstOutputConv = 20
    local secondOutputConv = 10
    local firstMaxPool = 4
    local secondMaxPool = 2

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    net:add(nn.View(math.floor((sizeData - firstInputConv + 1)/firstMaxPool)*firstOutputConv, 1))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv, 1))
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

    net:add(nn.TemporalConvolution(1, firstOutputConv, firstInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(firstMaxPool))
    local num1 = math.floor((sizeData - firstInputConv + 1)/firstMaxPool)*firstOutputConv
    net:add(nn.View(num1, 1))
    net:add(nn.TemporalConvolution(1, secondOutputConv, secondInputConv, 1))
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(secondMaxPool))
    local num2 = math.floor((num1 - secondInputConv + 1)/secondMaxPool)*secondOutputConv
    net:add(nn.View(num2, 1))
    -- net:add(nn.ReLU())
    net:add(nn.TemporalConvolution(1,1,1,1))
    local num = (  math.floor((math.floor((sizeData-firstInputConv+1)/firstMaxPool)*firstOutputConv - secondInputConv + 1)/secondMaxPool)*secondOutputConv  )
    -- net:add(nn.LookupTable(num,1))
    net:add(nn.ReLU())
    net:add(nn.View(num))
    net:add(nn.Linear(num, num*secondOutputConv))
    net:add(nn.ReLU())
    net:add(nn.Linear(num*secondOutputConv, nb_class))
    net:add(nn.ReLU())
    -- net:add(nn.LogSoftMax())

    id = 'conv(1,' .. firstInputConv .. ',' .. firstOutputConv .. ',1)-maxPool(' .. firstMaxPool ..
        ',1)-conv(1,' .. secondInputConv .. ',' .. secondOutputConv .. ',1)-maxPool(' .. secondMaxPool ..
        ',1)-conv(1,1,1,1)-' .. num .. '-732'

    print('MC-DCNN\n' .. net:__tostring())       -- print the model
end
