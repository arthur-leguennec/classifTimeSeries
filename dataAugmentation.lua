------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for data-augmentation functions ----------
------------------------------------------------------------------------------

require 'torch';
require 'cunn';
require 'math';


local function createData(data)
    local dataEx = data:clone()
    local size = (#data)[1]
    for i=1,size do
        local num = math.random() - 0.5
        if num < 0 then
            dataEx[i][1] = dataEx[i][1] * (1.005)
        elseif num > 0 then
            dataEx[i][1] = dataEx[i][1] * (0.995)
        else
            dataEx[i][1] = dataEx[i][1]
        end
    end
    return dataEx
end

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
        dataset.data[i] = createData(dataset.data[i])
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
    local stock = {{}}

    -- initialization of variables
    for k in pairs(classes) do
        nb_class = nb_class + 1
        num_classes[k] = 0
        num_classes_missing[k] = 0
    end

    -- how many label in each classes
    for i=1,dataset:size() do
        if not stock[dataset.label[i]] then
            stock[dataset.label[i]] = {}
        end
        num_classes[dataset.label[i]] = num_classes[dataset.label[i]] + 1
        table.insert(stock[dataset.label[i]], dataset.data[i])
    end

    -- what is the max label ?
    local max, label = 0, 0
    for k,v in pairs(num_classes) do
        if max<v then
            max = v
            label = k
        end
    end

    dataset.data = dataset.data:resize(max * nb_class, dataset:sizeData(), 1)
    dataset.label = dataset.label:resize(max * nb_class)

    -- how many by each classes
    for k,v in pairs(num_classes) do
        num_classes_missing[k] = max - v
    end

    local t = 1
    for k,v in pairs(num_classes_missing) do
        for i=1,v do
            local randomIndice = math.random(1,num_classes[k])
            local dataEx = createData(stock[k][randomIndice])
            table.insert(stock[k], dataEx)
            num_classes[k] = num_classes[k] + 1
            dataset.label[size+t] = k
            dataset.data[size+t] = dataEx
            t = t + 1
        end
    end

    print(dataset)
    return dataset
end
