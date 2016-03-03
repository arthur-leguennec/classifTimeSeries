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
local trainModelExist = false

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
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = learningRate
    trainer.maxIteration = maxIteration
    timer = torch.Timer()
    trainer:train(trainset)

    timer:stop()
    print(timer:time().real .. ' seconds for training the network.')
    timer:reset()
    if script then
        torch.save(fileTrain .. '_script_' .. model .. '_' .. learningRate .. '_' .. maxIteration .. '.t7', net)
    else
        torch.save(fileTrain .. '_' .. model .. '_' .. learningRate .. '_' .. maxIteration .. '.t7', net)
    end

    timer:stop()
end
