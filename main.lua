------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : main file for deep learning on time series ----
------------------------------------------------------------------------------

require 'nn';
require 'cunn';
require 'itorch';
require 'paths';
-- require 'cudnn';
-- cudnn.benchmark = true
-- cudnn.fastest = true
-- cudnn.verbose = true

require("conversionFileForTorch");
require("dataAugmentation");

print('\n')
print('+=================================================================+')
print('|+===============================================================+|')
print('||***************************************************************||')
print('||****-------------------------------------------------------****||')
print('||****----                                               ----****||')
print('||****--      Deep Learning on Time Series with torch      --****||')
print('||****----                                               ----****||')
print('||****-------------------------------------------------------****||')
print('||***************************************************************||')
print('|+===============================================================+|')
print('+=================================================================+\n\n')

cutorch.synchronizeAll()
timerGlobal = torch.Timer()
nb_class = 0
sizeData = 0
id = ''

-- -- -- -- -- 0. Command Line -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
dofile("cmdLine.lua")
print('Iter: ' .. maxIteration)
print('Learning rate: ' .. learningRate)
print('Learning rate decay: ' .. learningRateDecay .. '\n\n')

-- -- -- -- -- 1. Load and normalize data -- -- -- -- -- -- -- -- -- ---- -- --
trainset, classes = conversionCSV(fileTrain, mode_cuda)
for k in pairs(classes) do
    nb_class = nb_class + 1
end
if dataAug > 1 then
    trainset = dataAugmentationTimeSeries(trainset, dataAug)
end
-- trainset = dataAugmentationTimeSeriesPlus(trainset, classes)
testset, _ = conversionCSV(fileTest, mode_cuda)
print('\nThere are ' .. nb_class .. ' classes in this datasets.')
print('\n\n')

sizeData = trainset:sizeData()

-- -- -- -- -- -- 2. Define Neural Network -- -- -- -- -- -- -- -- -- -- -- -- --
dofile("model.lua")

printTitleModel()
net = nn.Sequential()
if model == 'leNet1' then
    neuralNetworkLenet1()
elseif model == 'neuralNetwork0' then
    neuralNetwork0()
elseif model == 'neuralNetwork1' then
    neuralNetwork1()
elseif model == 'leNet2' then
    neuralNetworkLenet2()
elseif model == 'mcdcnn' then
    neuralNetwork_MCDCNN()
else
    neuralNetworkLenet2()
end
print('\n' .. id)

if mode_cuda == true then
    net = net:cuda()
end

-- -- -- -- -- -- 3. Define Loss function  -- -- -- -- -- -- -- -- -- -- -- -- --
criterion = nn.CrossEntropyCriterion()
if mode_cuda == true then
    criterion = criterion:cuda()
end

-- -- -- -- -- -- 4. Train network on training data -- -- -- -- -- -- -- -- -- --
dofile("train.lua")

-- -- -- -- -- -- 5. Test network on test data   -- -- -- -- -- -- -- -- -- -- --
dofile("test.lua")

-- -- -- -- -- -- 6. The end   -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --s
timerGlobal:stop()
print('\n\nScript executed in ' .. timerGlobal:time().real .. ' secondes.')
