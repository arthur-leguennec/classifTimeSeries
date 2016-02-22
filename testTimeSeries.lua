------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : main file for deep learning on time series ----
------------------------------------------------------------------------------

require 'nn';
require 'cunn';
require 'itorch';
require 'paths';

require("model");
require("conversionFileForTorch");
require("dataAugmentation");

timerGlobal = torch.Timer()

-- -- -- -- -- 0. Command Line -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
dofile("cmdLine.lua")

-- -- -- -- -- 1. Load and normalize data -- -- -- -- -- -- -- -- -- ---- -- --
trainset, classes = conversionCSV(fileTrain, mode_cuda)
-- trainset = dataAugmentationTimeSeries(trainset, 5)
testset, _ = conversionCSV(fileTest, mode_cuda)
print('\nThere are ' .. #classes .. ' classes in this datasets.')
print('\n\n')

-- -- -- -- -- -- 2. Define Neural Network -- -- -- -- -- -- -- -- -- -- -- -- --
printTitleModel()
net = neuralNetworkLenet(net, trainset:sizeData(), #classes)
if mode_cuda == true then
    net = net:cuda()
end

-- -- -- -- -- -- 3. Define Loss function  -- -- -- -- -- -- -- -- -- -- -- -- --
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
-- criterion = nn.MSECriterion()   -- we choose the Mean Squared Error criterion
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
