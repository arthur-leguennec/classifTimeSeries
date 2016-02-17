require 'torch';
require 'cunn';
require 'math';

function dataAugmentationTimeSeries(dataset, z)
    -- Test for data augmentation
    local size = dataset:size()
    dataset.data = dataset.data:resize(size*z, 1, dataset:sizeData())
    dataset.label = dataset.label:resize(size*z)

    for i=1,(dataset:size()-size) do
        for j=1,dataset:sizeData() do
            num = math.random() - 0.5
            
            if num < 0 then
                dataset[i+size][1][j] = dataset[i][1][j] * (1.01)
            elseif num > 0 then
                dataset[i+size][1][j] = dataset[i][1][j] * (0.99)
            else
                dataset[i+size][1][j] = dataset[i][1][j]
            end
        end
    end
    return dataset
end
