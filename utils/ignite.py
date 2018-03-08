from collections import OrderedDict
from tqdm import tqdm

class progress_report(object):
    def __init__(self, epoch_length, prefix=None, progress_bar=True):
        self.epoch_length = epoch_length
        self.prefix = prefix
        self.progress_bar = progress_bar
        self.pbar = None
        
    def __call__(self, engine):
        prefix = ""
        if self.prefix is not None:
            prefix = "{}_".format(self.prefix)
        # Average metrics over the epoch thus far.
        metrics = OrderedDict(((prefix+'loss', 0),))
        iteration_in_epoch = (engine.current_iteration-1)%self.epoch_length+1
        idx_epoch_start = engine.current_iteration - iteration_in_epoch
        for iter_history in engine.history[idx_epoch_start:]:
            loss = iter_history[0]
            metrics[prefix+'loss'] += loss.item()
            for name, val in iter_history[-1].items():
                name = prefix+name
                if name not in metrics:
                    metrics[name] = 0
                metrics[name] += val
        for name in metrics:
            metrics[name] /= float(iteration_in_epoch)
                                    
        
        # Print to screen.
        desc = ""
        if hasattr(engine, 'current_epoch'):
            desc += "Epoch {}".format(engine.current_epoch)
        if self.progress_bar:
            if self.pbar is None:
                self.pbar = tqdm(total=self.epoch_length, desc=desc)
            self.pbar.set_postfix(metrics)
            self.pbar.update(1)
            if (engine.current_iteration)%self.epoch_length==0:
                self.pbar.close()
                self.pbar = None
        else:
            mstr = " ".join(["{}={:.3}".format(*x) for x in metrics.items()])
            desc = "{}: {}".format(desc, mstr) if len(desc) else mstr
            print(desc)
            
            
class metrics_handler(object):
    def __init__(self, measure_functions_dict=None):
        self.measure_functions = measure_functions_dict
        if measure_functions_dict is None:
            self.measure_functions = {}
            
    def __call__(self, output, target):
        measures = {}
        for m in self.measure_functions:
            measures[m] = self.measure_functions[m](output, target).data.item()
        return measures
        
    def add(self, metric_name, function):
        self.measure_functions[metric_name] = function 
