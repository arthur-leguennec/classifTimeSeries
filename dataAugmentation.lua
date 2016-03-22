------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for data-augmentation functions ----------
------------------------------------------------------------------------------

require 'torch';
require 'cunn';
require 'math';

function dataAugmentationTimeSeries(dataset, z)
    -- Test for data augmentation
    local size = dataset:size()
    dataset.data = dataset.data:resize(size*z, dataset:sizeData(), 1)
    dataset.label = dataset.label:resize(size*z)

    for i=size+1,z*size do
        if i % 5000 == 0 then
            print(i)
        end
        dataset.label[i] = dataset.label[i-size]
        for j=1,dataset:sizeData() do
            local num = math.random() - 0.5
            if num < 0 then
                dataset.data[i][j][1] = dataset.data[i-size][j][1] * (1.005)
            elseif num > 0 then
                dataset.data[i][j][1] = dataset.data[i-size][j][1] * (0.995)
            else
                dataset.data[i][j][1] = dataset.data[i-size][j][1]
            end
        end
    end
    print(dataset)
    return dataset
end

-------------------------------------------------------------------------------------------------
--------- TODO augmented the dataset for have a same number of examples in each classes ---------
-------------------------------------------------------------------------------------------------
function dataAugmentationTimeSeriesPlus(dataset, classes)
    local size = dataset:size()
    local nb_class = 0
    local num_classes = {}
    local num_classes_missing = {}

    -- initialization of variables
    for k in pairs(classes) do
        nb_class = nb_class + 1
        num_classes[k] = 0
        num_classes_missing[k] = 0
    end

    -- how many label in each classes
    for i=1,dataset:size() do
        num_classes[dataset[i][2]] = num_classes[dataset[i][2]] + 1
    end

    -- what is the max label ?
    local max, label = 0, 0
    for k,v in pairs(num_classes) do
        if max<v then
            max = v
            label = k
        end
    end

    -- how many by each classes
    for k,v in pairs(num_classes) do
        num_classes_missing[k] = max - v
    end

    for k,v in pairs(num_classes_missing) do
        for i=1,v do
            
        end
    end
end


local function createData(data)
    local dataEx = data:clone()
    local size = #data
    for i=1,size do
        local num = math.random() - 0.5
        if num < 0 then
            dataEx[i] = dataEx[i] * (1.005)
        elseif num > 0 then
            dataEx[i] = dataEx[i] * (0.995)
        else
            dataEx[i] = dataEx[i]
        end
    end
    return dataEx
end
