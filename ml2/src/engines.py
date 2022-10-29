# engines

import torch
from torchmetrics.aggregation import MeanMetric

class DefaultTrainEvaluate:
    """
    DefaultTrainEvaluate Class
    
    Default form
    """
    @staticmethod
    def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
        model.train()
        loss_mean = MeanMetric()
        metric_mean = MeanMetric()
        
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs.reshape((-1, outputs.shape[-1]))
            targets = targets.reshape((-1,))
            loss = loss_fn(outputs, targets)
            metric = metric_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_mean.update(loss.to('cpu'))
            metric_mean.update(metric.to('cpu'))

            scheduler.step()

        summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

        return summary
    
    @staticmethod
    def evaluate(loader, model, loss_fn, metric_fn, device):
        model.eval()
        loss_mean = MeanMetric()
        metric_mean = MeanMetric()
        
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            outputs = outputs.reshape((-1, outputs.shape[-1]))
            targets = targets.reshape((-1,))
            loss = loss_fn(outputs, targets)
            metric = metric_fn(outputs, targets)

            loss_mean.update(loss.to('cpu'))
            metric_mean.update(metric.to('cpu'))
        
        summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

        return summary
        
class Summary1(DefaultTrainEvaluate):
    pass

class Test(DefaultTrainEvaluate):
    pass