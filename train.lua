------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for training the model -------------------
------------------------------------------------------------------------------

print('\n--------------------------------------------------------------')
print('-------------------------- Training --------------------------')
print('--------------------------------------------------------------')

local answer
local pathTrainModelLoad
local pathTrainModelSave
local trainModel = false

local function training(params)
    local iteration = 1
    local maxIteration = params.maxIteration or 50
    local momentum = params.momentum or 0.05
    local learningRateDecay = params.learningRateDecay or 0.00005
    local currentLearningRate = params.learningRate or 0.1
    local minError = params.minError or 0.00001
    local module = params.module
    local criterion = params.criterion
    local verbose = params.verbose and true
    local dataset = params.dataset
    local mini_batch_size = params.miniBatchSize or 1
    local i = 0
    local t

    local listCurrentError = {}

    print("# StochasticGradientEx: training")

    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')

    while true do
        local currentError = 0

        t = 1
        while t <= (dataset:size()/mini_batch_size) do
            if t+i > dataset:size() then
                i = 0
            end
            local example = dataset[shuffledIndices[t]]
            local input = example[1]
            local target = example[2]

            currentError = currentError + criterion:forward(module:forward(input), target)

            module:updateGradInput(input, criterion:updateGradInput(module.output, target))
            module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
            t = t + 1
        end
        i = i + mini_batch_size

        currentError = currentError * mini_batch_size / dataset:size()

        if verbose then
            print("# current error = " .. currentError)
        end

        listCurrentError[iteration] = currentError
        iteration = iteration + 1
        currentLearningRate = learningRate/(1+iteration*learningRateDecay)

        if maxIteration > 0 and iteration > maxIteration then
            print("# StochasticGradientEx: you have reached the maximum number of iterations")
            print("# training error = " .. currentError)
            trainingError = currentError
            break
        end
    end

    if #listCurrentError > 0 then
        return listCurrentError
    else
        return nil
    end
end

net:training()

if not script then
    io.write('load a train model (y/[n])? ')
    io.flush()
    answer = io.read()

    if answer == 'y' or answer == 'yes' or answer =='Y' then
        io.write('path file? ')
        io.flush()
        pathTrainModelLoad = io.read()
        net = torch.load(pathTrainModelLoad)
        trainModel = true
        io.write('retrain this model (y/[n])? ')
        io.flush()
        answer = io.read()
        if answer == 'y' or answer == 'yes' or answer =='Y' then
            trainModel = false
        end
    end
end

if trainModel == false then
    local params = {}
    params.learningRate = learningRate
    params.maxIteration = 20
    params.verbose = false
    params.module = net
    params.criterion = criterion
    params.momentum = momentum
    params.learningRateDecay = learningRateDecay
    params.dataset = trainset
    params.miniBatchSize = miniBatchSize

    local listCurrentError = {}

    local listError = {}
    local seedCPU
    local seedGPU
    if mode_cuda then
        seedGPU = cutorch.seedAll()
        seedCPU = torch.seed()
        cutorch.manualSeedAll(seedGPU)
        torch.manualSeed(seedCPU)
        print("\n[CUDA] The seed is: " .. cutorch.initialSeed())
        print("[CPU] The seed is: " .. torch.initialSeed())
    else
        seedCPU = torch.seed()
        torch.manualSeed(seedCPU)
        print("\n[NO CUDA] The seed is: " .. torch.initialSeed())
    end
    local seedShuffledIndices = torch.getRNGState()

    for i=1,7 do
        net:zeroGradParameters()
        net:reset()
        torch.setRNGState(seedShuffledIndices)
        listCurrentError[i] = training(params)
        torch.save('test.t7', net)
        listError[listCurrentError[i][#listCurrentError[i]]] = net:clone()
    end

    local minError = next(listError)
    local optimNet = listError[minError]
    for k,v in pairs(listError) do
        if k < minError then
            minError, optimNet = k, v
        end
    end
    listError = nil

    print("\nWe choose the initialization with the smallest error.")
    print("So, we choose " .. minError)

    net = optimNet:clone()
    params.module = net
    params.maxIteration = maxIteration
    params.verbose = true
    timer = torch.Timer()
    torch.setRNGState(seedShuffledIndices)
    listCurrentError = training(params)

    timer:stop()
    print(timer:time().real .. ' seconds for training the network.')
    timer:reset()

    timer:stop()
end
