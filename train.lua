------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for training the model -------------------
------------------------------------------------------------------------------


function printTitleTrain()
    print('\n--------------------------------------------------------------')
    print('-------------------------- Training --------------------------')
    print('--------------------------------------------------------------')
end

function train(net, criterion, learningRate, maxIteration)
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = learningRate
    trainer.maxIteration = maxIteration    -- just do 5 epochs of training
    timer = torch.Timer()
    trainer:train(trainset)

    timer:stop()
    print(timer:time().real .. ' seconds for training the network.')
    timer:reset()
end
