------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : main file for deep learning on time series ----
------------------------------------------------------------------------------

require 'nn';
require 'cunn';
require 'itorch';
require 'paths';

require("conversionFileForTorch");
require("dataAugmentation");

print('\n')
print('+===============================================================+')
print('|***************************************************************|')
print('|****-------------------------------------------------------****|')
print('|****----                                               ----****|')
print('|****--      Deep Learning on Time Series with torch      --****|')
print('|****----                                               ----****|')
print('|****-------------------------------------------------------****|')
print('|***************************************************************|')
print('+===============================================================+\n\n')

timerGlobal = torch.Timer()
nb_class = 0
sizeData = 0

-- -- -- -- -- 0. Command Line -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
dofile("cmdLine.lua")

-- -- -- -- -- 1. Load and normalize data -- -- -- -- -- -- -- -- -- ---- -- --
trainset, classes = conversionCSV(fileTrain, mode_cuda)
for k in pairs(classes) do
    nb_class = nb_class + 1
end
-- trainset = dataAugmentationTimeSeries(trainset, 10)
-- trainset = shuffledDataset(trainset)
testset, _ = conversionCSV(fileTest, mode_cuda)
print('\nThere are ' .. nb_class .. ' classes in this datasets.')
print('\n\n')

sizeData = trainset:sizeData()

-- -- -- -- -- -- 2. Define Neural Network -- -- -- -- -- -- -- -- -- -- -- -- --
dofile("model.lua")

printTitleModel()
net = nn.Sequential()
neuralNetworkLenet()
if mode_cuda == true then
    net = net:cuda()
end

-- -- -- -- -- -- 3. Define Loss function  -- -- -- -- -- -- -- -- -- -- -- -- --
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
-- criterion = nn.MSECriterion()   -- we choose the Mean Squared Error criterion
-- criterion = nn.MarginCriterion()
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
