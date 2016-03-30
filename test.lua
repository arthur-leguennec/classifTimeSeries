------------------------------------------------------------------------------
---------------- Creation file : 02/18/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for testing the model --------------------
------------------------------------------------------------------------------

require("saveResult");

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

accuracy = correct/testset:size()
print(correct .. ' right answers, so', 100*accuracy .. '% accuracy in ' .. timer:time().real .. ' seconds.')

if not script then
    io.write('save this model ([y]/n)? ')
    io.flush()
    answer=io.read()
    if answer ~= 'n' and answer ~= 'N' and answer ~= 'no' then
        io.write('where? (default ' .. params.pathData .. paths.basename(params.pathData) .. '.t7) ')
        io.flush()
        pathTrainModelSave = io.read()
        if pathTrainModelSave ~= '' then
            torch.save(pathTrainModelSave, net)
        else
            torch.save(params.pathData .. paths.basename(params.pathData) .. '.t7', net)
        end
    end
end

if script then
    addResult(saveFile, model, id, learningRate, learningRateDecay, maxIteration, dataAug, accuracy)
end

if not script then
    io.write('save the result ([y]/n)? ')
    io.flush()
    answer = io.read()
    if answer ~= 'n' and answer ~= 'N' and answer ~= 'no' then
        local yolo = 'iter, ' .. maxIteration .. '\nlearning_rate, ' .. learningRate .. '\nerror_rate, ' .. 100*accuracy
        torch.save(params.pathData .. paths.basename(params.pathData) .. '_' .. maxIteration .. '_iter', yolo)
    end
end
