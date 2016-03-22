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
cmd:option('-lrd', 0, 'learning rate decay')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-iter', 15, 'max iteration for training')
cmd:option('-saveFile', '', 'path file for save the results')
cmd:option('-model', 'leNet1', 'choose the existing model')
cmd:option('-dataAugmentation', 1, 'factor of data augmentation')
cmd:option('-script', false, 'if you run a script for testing several things')
paramsCmd = cmd:parse(arg)

if paramsCmd.pathData == '' then
    if paramsCmd.fileTrain == '' or paramsCmd.fileTest == '' then
        fileTrain = PATH_UCR .. '/ECG5000/ECG5000_TRAIN'
        fileTest = PATH_UCR .. '/ECG5000/ECG5000_TEST'
        saveFile = PATH_UCR .. './ECG5000/ECG5000_result'
    else
        fileTrain = paramsCmd.fileTrain
        fileTest = paramsCmd.fileTest
        saveFile = paramsCmd.saveFile
    end
else
    fileTrain = paramsCmd.pathData .. paths.basename(paramsCmd.pathData) .. '_TRAIN'
    fileTest = paramsCmd.pathData .. paths.basename(paramsCmd.pathData) .. '_TEST'
    saveFile = paramsCmd.saveFile .. paths.basename(paramsCmd.saveFile) .. '_result'
end

mode_cuda = not paramsCmd.noModeCuda
learningRate = paramsCmd.lr
learningRateDecay = paramsCmd.lrd
maxIteration = paramsCmd.iter
model = paramsCmd.model
script = paramsCmd.script
dataAug = paramsCmd.dataAugmentation
momentum = paramsCmd.momentum
