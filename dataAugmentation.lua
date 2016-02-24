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
            num = math.random() - 0.5
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
-------- TODO augmented the dataset for to have a same number of examples in each classes--------
-------------------------------------------------------------------------------------------------
function dataAugmentationTimeSeriesPlus(dataset, classes, z)
    local size = dataset:size()
    local nb_class = 0
    local num_ex_classes_train = {}

    for k in pairs(classes) do
        nb_class = nb_class + 1
        nb_ex_classes_train[k] = 0
    end

    for i=1,dataset:size() do
        local groundtruth = trainset.label[i]
        num_ex_classes_train[groundtruth] = num_ex_classes_train[groundtruth] + 1
    end
end
