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

io.write('load a train model (y/n)? (default n)')
io.flush()
answer = io.read()

if answer == 'y' then
    io.write('path file? ')
    io.flush()
    pathTrainModelLoad = io.read()
    net = torch.load(pathTrainModelLoad)
    trainModel = true
    io.write('retrain this model (y/n)? (default n)')
    io.flush()
    answer = io.read()
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

    if answer ~= 'n' and answer ~= 'N' and answer ~= 'no' then
        io.write('where? (default ' .. params.pathData .. paths.basename(params.pathData) .. '.t7)')
        io.flush()
        pathTrainModelSave = io.read()
        if pathTrainModelSave ~= '' then
            torch.save(pathTrainModelSave, net)
        else
            torch.save(params.pathData .. paths.basename(params.pathData) .. '.t7', net)
        end
    end
end
