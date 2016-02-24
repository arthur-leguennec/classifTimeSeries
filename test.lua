------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for testing the model --------------------
------------------------------------------------------------------------------

print('\n--------------------------------------------------------------')
print('-------------------------- Testing ---------------------------')
print('--------------------------------------------------------------')


class_performance = {}
num_ex_classes = {}
num_ex_classes_train = {}

for k in pairs(classes) do
    class_performance[k] = 0
    num_ex_classes[k] = 0
    num_ex_classes_train[k] = 0
end

for i=1,trainset:size() do
    local groundtruth = trainset.label[i]
    num_ex_classes_train[groundtruth] = num_ex_classes_train[groundtruth] + 1
end

net:evaluate()

correct = 0
timer = torch.Timer()
timer:resume()
for i=1,testset:size() do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
    num_ex_classes[groundtruth] = num_ex_classes[groundtruth] + 1
end
timer:stop()
print('')

for k in pairs(classes) do
    print('+-------------------------------------------------------+')
    print('|\t\t\t Class ' .. classes[k] .. '\t\t\t|')
    print('|=======================================================|')
    print('| Good answers: ' .. class_performance[k], '\t\t\t\t|')
    print('| Number of data test: ' .. num_ex_classes[k], '\t\t\t|')
    print('| Number of data train: ' .. num_ex_classes_train[k], '\t\t\t|')
    print('| Percentage of accuracy: ' .. string.format("%.2f", class_performance[k]*100/num_ex_classes[k]) .. ' %\t\t\t|')
    print('+-------------------------------------------------------+')
    print('')
end

print(correct .. ' right answers, so', 100*correct/testset:size() .. '% accuracy in ' .. timer:time().real .. ' seconds.')
