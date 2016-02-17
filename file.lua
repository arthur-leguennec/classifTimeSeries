require 'nn';

net = nn.Sequential()
net:add(nn.SpatialConvolution(1,6,5,5))     -- 1 input image channels, 6 outputs channels, 5x5 convolution kernel
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

print('Lenet5\n' .. net:__tostring())

input = torch.rand(1, 32, 32)
output = net:forward(input)
print(output)
net:zeroGradParameters()
gradInput = net:backward(input, torch.rand(10))


criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
criterion:forward(output, 3) -- let's say the groundtruth was class number: 3
gradients = criterion:backward(output, 3)
gradInput = net:backward(input, gradients)

m = nn.SpatialConvolution(1,3,2,2) -- learn 3 2x2 kernels
print(m.weight) -- initially, the weights are randomly initialized

print(m.bias) -- The operation in a convolution layer is: output = convolution(input,weight) + bias
