import flgo.experiment.analyzer
task='./my_task'
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedpac', 'fedala','fedrod','pfedme']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)