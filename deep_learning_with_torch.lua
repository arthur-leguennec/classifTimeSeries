-- Exercice of the page "https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb"

require 'nn';
require 'image';
require 'cunn';


-- -- -- -- -- 1. Load and normalize data -- -- -- -- ---- -- -- -- ---- -- --
-- Download and unzip the file
-- os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip --no-check-certificate')
-- os.execute('unzip cifar10torchsmall.zip')

trainset = torch.load('cifar10-train.t7')   -- Training data
testset = torch.load('cifar10-test.t7')     -- Testing data
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}    -- Defines classes (10 here)
print(trainset)

setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);
trainset.data = trainset.data:double():cuda()  -- convert the data from a ByteTensor to a DoubleTensor.

print(#trainset)

function trainset:size()
    return self.data:size(1)
end


mean = {}   -- store the mean, to normalize the test set in the future
stdv = {}   -- store the standart-deviation for the future

for i=1,3 do -- over each image channel (r, g, b)
    mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])    -- mean subtraction
    stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()  -- std estimation
    print('Channel ' .. i .. ', Standart Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])     -- std scaling
end


-- -- -- -- -- 2. Define Neural Network -- -- -- -- -- -- -- -- -- -- -- -- --
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,5,5))     -- 3 input image channels, 6 outputs channels, 5x5 convolution kernel
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max
net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between inputs and weights)
net:add(nn.ReLU())
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())
net:add(nn.Linear(84, 10))                  -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                    -- converts the output to a log-probability. Useful for classification problems
net = net:cuda()

print('Lenet5\n' .. net:__tostring())       -- print the model

-- print(net.modules)

-- -- -- -- -- 3. Define Loss function  -- -- -- -- -- -- -- -- -- -- -- -- --
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
criterion = criterion:cuda()

-- -- -- -- -- 4. Train network on training data -- -- -- -- -- -- -- -- -- --
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.002
trainer.maxIteration = 5    -- just do 5 epochs of training
trainer:train(trainset)

-- print(net.modules)

-- -- -- -- -- 5. Test network on test data   -- -- -- -- -- -- -- -- -- -- --
testset.data = testset.data:double():cuda()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {} }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {} }]:div(stdv[i]) -- std scaling
end

print(classes[testset.label[100]])
predicted = net:forward(testset.data[100])
print(predicted:exp())

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. '%')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end
