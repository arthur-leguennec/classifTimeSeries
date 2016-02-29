------------------------------------------------------------------------------
---------------- Creation file : 02/26/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for saving the result about parameters ---
------------------------------------------------------------------------------

require 'csvigo'
require 'paths'

local pathFileResult = ''

function addResult(pathData, id, learningRate, maxIter, accuracy)
    pathFileResult = pathData .. paths.basename(pathData) .. '_result'
    if paths.filep(pathFileResult) == false then
        csvigo.save({path = pathFileResult, data = {}})
    end

    local fileResult = csvigo.load({path=pathFileResult, mode='raw'})

    fileResult[1] = {'id', 'learning rate', 'max iteration', 'result', 'error rate'}
    fileResult[#fileResult + 1] = {id, learningRate, maxIter, accuracy, 1-accuracy}
    csvigo.save({path = pathFileResult, data = fileResult})
end