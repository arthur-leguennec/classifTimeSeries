require 'nn';
require 'image';
require 'cunn';
require 'itorch';
require 'paths';

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
    fileTrain = params.fileTrain
    fileTest = params.fileTest
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
trainset = dataAugmentationTimeSeries(trainset, 10)
testset, _ = conversionCSV(fileTest, mode_cuda)
testset = normalizeDataset(testset)
print('\n\n')

-- -- -- -- -- -- 2. Define Neural Network -- -- -- -- -- -- -- -- -- -- -- -- --
net = nn.Sequential()
-- net:add(nn.TemporalConvolution(5, 1, 3))
-- net:add(nn.ReLU())
-- net:add(nn.View(136))
-- net:add(nn.Linear(trainset:sizeData(), 5000))        -- fully connected layer (matrix multiplication between inputs and weights)
-- net:add(nn.ReLU())
net:add(nn.Linear(trainset:sizeData(), 500))        -- fully connected layer (matrix multiplication between inputs and weights)
net:add(nn.ReLU())
net:add(nn.Linear(500, #classes))
net:add(nn.LogSoftMax())
if mode_cuda == true then
    net = net:cuda()
end

print('Neural Network test\n' .. net:__tostring())       -- print the model

-- -- -- -- -- -- 3. Define Loss function  -- -- -- -- -- -- -- -- -- -- -- -- --
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
-- criterion = nn.MSECriterion()   -- we choose the Mean Squared Error criterion
if mode_cuda == true then
    criterion = criterion:cuda()
end

-- -- -- -- -- -- 4. Train network on training data -- -- -- -- -- -- -- -- -- --
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = learningRate
trainer.maxIteration = maxIteration    -- just do 5 epochs of training
timer = torch.Timer()
trainer:train(trainset)

-- print(net.modules

-- for i = 1,trainset:size() do
--     -- random sample
--     local input = trainset.data[i][1];     -- normally distributed example in 2d
--     local output = trainset.label[i];
--
--     -- feed it to the neural network and the criterion
--     net:forward(input)
--     criterion:forward(net:forward(input), output)
--
--     -- train over this example in 3 steps
--     -- (1) zero the accumulation of the gradients
--     net:zeroGradParameters()
--     -- (2) accumulate gradients
--     net:backward(input, criterion:backward(net.output, output))
--     -- (3) update parameters with a 0.01 learning rate
--     net:updateParameters(0.02)
-- end


timer:stop()
print(timer:time().real .. ' seconds for training the network.')
timer:reset()

-- -- -- -- -- -- 5. Test network on test data   -- -- -- -- -- -- -- -- -- -- --
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
    local prediction = net:forward(testset.data[i][1])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
    num_ex_classes[groundtruth] = num_ex_classes[groundtruth] + 1
end
timer:stop()
print('\n')
print(correct .. ' right answers, so', 100*correct/testset:size() .. '% accuracy in ' .. timer:time().real .. ' seconds.')

for k in pairs(classes) do
    print('Class ' .. classes[k])
    print('Good answers:', class_performance[k])
    print('Number of data test:', num_ex_classes[k])
    print('Number of data train:', num_ex_classes_train[k])
    print('')
end

timerGlobal:stop()
print('')
print('Script executed in ' .. timerGlobal:time().real .. ' secondes.')
