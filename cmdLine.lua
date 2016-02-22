------------------------------------------------------------------------------
---------------- Creation file : 02/22/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for training the model -------------------
------------------------------------------------------------------------------

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
    if params.fileTrain == '' or params.fileTest == '' then
        fileTrain = '/home/arthur/Downloads/UCR_TS_Archive_2015/ECG5000/ECG5000_TRAIN'
        fileTest = '/home/arthur/Downloads/UCR_TS_Archive_2015/ECG5000/ECG5000_TEST'
    else
        fileTrain = params.fileTrain
        fileTest = params.fileTest
    end
else
    fileTrain = params.pathData .. paths.basename(params.pathData) .. '_TRAIN'
    fileTest = params.pathData .. paths.basename(params.pathData) .. '_TEST'
end

mode_cuda = not params.noModeCuda
learningRate = params.lr
maxIteration = params.iter
