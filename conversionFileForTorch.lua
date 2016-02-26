------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for conversion the data for torch --------
------------------------------------------------------------------------------


require 'csvigo';
require 'torch';
require 'cunn';


function conversionCSV(filename, mode_cuda)
    --- Converts CVS file to a tensor for torch.
    --- Return a tensor and a table of classes
    --- Warning: Works only with the UCR datasets (see http://www.cs.ucr.edu/~eamonn/time_series_data/).
    local mode_cuda = mode_cuda or false
    local datasetcsv = csvigo.load({path=filename, mode='raw'})
    local dataset = {}
    local classes = {}

    if mode_cuda ~= true then
        mode_cuda = false
    else
        mode_cuda = true
    end

    function dataset:size()
        return #datasetcsv
    end
    function dataset:sizeData()
        return  #datasetcsv[1] - 1
    end

    local data = torch.DoubleTensor(dataset:size(), dataset:sizeData(), 1)
    local label = torch.CharTensor(dataset:size())

    for i=1,dataset:size() do
        for j=2,#datasetcsv[i] do
            data[i][j-1] = datasetcsv[i][j]
        end
        label[i] = datasetcsv[i][1]

        classes[label[i]] = label[i]
    end

    local tabTmp = {}
    local i = 0

    for k in pairs(classes) do
        i = i + 1
        tabTmp[k] = i
    end

    classes = {}
    for i=1,dataset:size() do
        label[i] = tabTmp[label[i]]
        classes[label[i]] = label[i]
    end

    if mode_cuda == true then
        dataset.data = data:cuda()
        dataset.label = label:cuda()
    else
        dataset.data = data
        dataset.label = label
    end

    setmetatable(dataset,
        {__index = function(t, i)
                        return {t.data[i], t.label[i]}
                    end}
    );

    -- print(dataset)
    -- print(classes)

    function dataset:size()
        return self.data:size(1)
    end

    return dataset, classes
end


function shuffledDataset(dataset)
    y = torch.randperm(dataset:size())
    local datasetTmp = {}

    for k in pairs(dataset) do
        datasetTmp[k] = dataset[k]
    end

    for i=1,dataset:size() do
        dataset[i] = datasetTmp[y[i]]
    end

    return dataset
end


function normalizeDataset(dataset)
    local mean = 0   -- store the mean, to normalize the test set in the future
    local stdv = 0   -- store the standart-deviation for the future

    mean = dataset.data[{ {}, {}, {} }]:mean() -- mean estimation
    print('Mean: ' .. mean)
    dataset.data[{ {}, {}, {} }]:add(-mean)    -- mean subtraction
    stdv = dataset.data[{ {}, {}, {} }]:std()  -- std estimation
    print('Standart Deviation: ' .. stdv)
    dataset.data[{ {}, {}, {} }]:div(stdv)     -- std scaling
    return dataset
end
