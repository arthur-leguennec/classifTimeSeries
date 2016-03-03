------------------------------------------------------------------------------
---------------- Creation file : 02/23/2016 ----------------------------------
---------------- Author : Arthur Le Guennec ----------------------------------
---------------- Description : file for show the graph of training and testing
------------------------------------------------------------------------------

require 'gnuplot'

VisualizationVector = {}

function VisualizationVector:new()
    o = {
        data = torch.Tensor(1e2):zero(),
        sampleCount = 0
    }
    setmetatable(o, self)
    self.__index = self
    return o
end


function VisualizationVector:addSample(sample)
    local dataSize = (#self.data)[1]
    if self.sampleCount + 1 >= dataSize then
        self.data:resize(dataSize+50)
        self.data:sub(dataSize+1, dataSize+50):zero()
    end
    self.sampleCount = self.sampleCount + 1
    self.data[self.sampleCount] = sample
end

loss_samples = VisualizationVector:new()
cv_loss_samples = VisualizationVector:new()

function setup_loss_graph(sgd_params)
    gnuplot.title('learning rate: ' .. sgd_params.learningRate ..
                ' decay: ' .. sgd_params.learningRateDecay ..
                ' weightDecay: ' .. sgd_params.weightDecay ..
                ' momentum: ' .. sgd_params.momentum)
    gnuplot.grid(true)
end

function visualize_loss(loss, cv_loss)
    loss_samples:addSample(loss)
    cv_loss_samples:addSample(cv_loss)

    gnuplot.plot({'loss', loss_samples.data, '-'}, {'cv', cv_loss_samples.data, '-'})
end
