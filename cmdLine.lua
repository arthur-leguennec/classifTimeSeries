------------------------------------------------------------------------------
---------------- Creation file : 02/22/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for the command line ---------------------
------------------------------------------------------------------------------

PATH_UCR = '../UCR_TS_Archive_2015'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-pathData', '', 'path to file with data')
cmd:option('-fileTrain', '', 'filename for training')
cmd:option('-fileTest', '', 'filename for testing')
cmd:option('-noModeCuda', false, 'activate the mode cuda')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-iter', 15, 'max iteration for training')
cmd:option('-saveFile', '', 'path file for save the results')
cmd:option('-model', 'leNet', 'choose the existing model')
cmd:option('-script', false, 'if you run a script for testing several things')
params = cmd:parse(arg)

if params.pathData == '' then
    if params.fileTrain == '' or params.fileTest == '' then
        fileTrain = PATH_UCR .. '/ECG5000/ECG5000_TRAIN'
        fileTest = PATH_UCR .. '/ECG5000/ECG5000_TEST'
        saveFile = PATH_UCR .. './ECG5000/ECG5000_result'
    else
        fileTrain = params.fileTrain
        fileTest = params.fileTest
        saveFile = params.saveFile
    end
else
    fileTrain = params.pathData .. paths.basename(params.pathData) .. '_TRAIN'
    fileTest = params.pathData .. paths.basename(params.pathData) .. '_TEST'
    saveFile = params.saveFile .. paths.basename(params.saveFile) .. '_result'
end

mode_cuda = not params.noModeCuda
learningRate = params.lr
maxIteration = params.iter
model = params.model
script = params.script
