------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for training the model -------------------
------------------------------------------------------------------------------


print('\n--------------------------------------------------------------')
print('-------------------------- Training --------------------------')
print('--------------------------------------------------------------')

local answer
local pathTrainModel
local trainModel = false

io.write('load a train model (y/n)? (default n)')
io.flush()
answer = io.read()

if answer == 'y' then
    io.write('path file? ')
    io.flush()
    pathTrainModel = io.read()
    net = torch.load(pathTrainModel)
    trainModel = true
end

io.write('retrain this model (y/n)? (default n)')
io.flush()
answer = io.read()

if answer == 'y' then
    trainModel = false
end

if trainModel == false then
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = learningRate
    trainer.maxIteration = maxIteration    -- just do 5 epochs of training
    timer = torch.Timer()
    trainer:train(trainset)

    timer:stop()
    print(timer:time().real .. ' seconds for training the network.')
    timer:reset()

    timer:stop()

    io.write('save this train model (y/n)? (default y)')
    io.flush()
    answer=io.read()

    if answer ~= 'n' or answer ~= 'N' or answer ~= 'no' then
        io.write('where? (default test.t7 in this file)')
        io.flush()
        pathTrainModel = io.read()
        if pathTrainModel ~= '' then
            torch.save(pathTrainModel, net)
        else
            torch.save('./test.t7', net)
        end
    end
end
