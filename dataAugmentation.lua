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
