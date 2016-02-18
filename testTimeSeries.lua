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
require("train");
require("test");
require("conversionFileForTorch");
require("dataAugmentation");

timerGlobal = torch.Timer()


-- -- -- -- -- 0. Command Line -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-pathData', '', 'path to file with data')
cmd:option('-fileTrain', '', 'filename for training')
cmd:option('-fileTest', '', 'filename for testing')
cmd:option('-noModeCuda', false, 'activate the mode cuda')
cmd:option('-lr', 0.02, 'learning rate')
cmd:option('-iter', 5, 'max iteration for training')
params = cmd:parse(arg)

if params.pathData == '' then
    if params.fileTrain == '' or params.fileTest == '' then
        fileTrain = '/home/arthur/Downloads/UCR_TS_Archive_2015/ECG5000/ECG5000_TRAIN'
        fileTest = '/home/arthur/Downloads/UCR_TS_Archive_2015/ECG5000/ECG5000_TEST'
    else
        fileTrain = params.fileTrain
        fileTest = params.fileTest
    end
else
    fileTrain = params.pathData .. paths.basename(params.pathData) .. '_TRAIN'
    fileTest = params.pathData .. paths.basename(params.pathData) .. '_TEST'
end

mode_cuda = not params.noModeCuda
learningRate = params.lr
maxIteration = params.iter



-- -- -- -- -- 1. Load and normalize data -- -- -- -- -- -- -- -- -- ---- -- --
trainset, classes = conversionCSV(fileTrain, mode_cuda)
trainset = normalizeDataset(trainset)
-- trainset = dataAugmentationTimeSeries(trainset, 5)
testset, _ = conversionCSV(fileTest, mode_cuda)
testset = normalizeDataset(testset)
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
printTitleTrain()

train(net, criterion, learningRate, maxIteration)

-- -- -- -- -- -- 5. Test network on test data   -- -- -- -- -- -- -- -- -- -- --
printTitleTest()
class_performance = {}
num_ex_classes = {}
num_ex_classes_train = {}

for k in pairs(classes) do
    class_performance[k] = 0
    num_ex_classes[k] = 0
    num_ex_classes_train[k] = 0
end

for i=1,trainset:size() do
    local groundtruth = trainset.label[i]
    num_ex_classes_train[groundtruth] = num_ex_classes_train[groundtruth] + 1
end

correct = 0
timer:resume()
for i=1,testset:size() do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
    num_ex_classes[groundtruth] = num_ex_classes[groundtruth] + 1
end
timer:stop()
print('')

for k in pairs(classes) do
    print('+-------------------------------------------------------+')
    print('|\t\t\t Class ' .. classes[k] .. '\t\t\t|')
    print('|=======================================================|')
    print('| Good answers: ' .. class_performance[k], '\t\t\t\t|')
    print('| Number of data test: ' .. num_ex_classes[k], '\t\t\t|')
    print('| Number of data train: ' .. num_ex_classes_train[k], '\t\t\t|')
    print('| Percentage of accuracy: ' .. string.format("%.2f", class_performance[k]*100/num_ex_classes[k]) .. ' %\t\t\t|')
    print('+-------------------------------------------------------+')
    print('')
end

print(correct .. ' right answers, so', 100*correct/testset:size() .. '% accuracy in ' .. timer:time().real .. ' seconds.')

timerGlobal:stop()
print('\n\nScript executed in ' .. timerGlobal:time().real .. ' secondes.')
