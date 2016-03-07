------------------------------------------------------------------------------
---------------- Creation file : 02/26/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for saving the result about parameters ---
------------------------------------------------------------------------------

require 'csvigo'
require 'paths'

local pathFileResult = ''

function addResult(pathData, model, id, learningRate, learningRateDecay, maxIter, dataAug, accuracy)
    pathFileResult = pathData .. paths.basename(pathData) .. '_result'
    if paths.filep(pathFileResult) == false then
        csvigo.save({path = pathFileResult, data = {}})
    end

    local fileResult = csvigo.load({path=pathFileResult, mode='raw'})

    fileResult[1] = {'model', 'id', 'learning rate', 'learning rate decay', 'max iteration', 'data augmentation', 'result', 'error rate'}
    fileResult[#fileResult + 1] = {model, id, learningRate, learningRateDecay, maxIter, dataAug, accuracy, 1-accuracy}
    csvigo.save({path = pathFileResult, data = fileResult})

    -- local fileUCR = csvigo.load({path='./UCR_Data_results.csv', mode='raw'})
    --
    -- local tmp = 1
    -- for k in pairs(fileUCR[1]) do
    --     tmp = tmp + 1
    --     if k == model then
    --         break
    --     else
    --         fileUCR[1][#fileUCR[1]+1] = model
    --     end
    -- end
    --
    -- for k,v in pairs(fileUCR) do
    --     if v[1] == paths.basename(pathData) then
    --         v[tmp] = 1-accuracy
    --         print('yolo')
    --         break
    --     end
    -- end
    -- csvigo.save({path = './UCR_Data_results.csv', data = fileUCR})
end
