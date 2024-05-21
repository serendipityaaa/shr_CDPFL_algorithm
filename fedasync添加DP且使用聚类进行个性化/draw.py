import flgo.experiment.analyzer
task = './my_task1'
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedasync']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'xlabel':'communication_rounds', 'ylabel':'Loss','title':'Loss function on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'xlabel':'communication_rounds', 'ylabel':'accuracy','title':'accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)