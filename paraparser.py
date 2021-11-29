import argparse
from texttable import Texttable

def parameter_parser_clustering():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a good quality representation without grid search.
    The experimental results are obtained by setting following default hyperparameters
    """

    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--dataset-id",
                    type = int,
                default = 1)
    
    parser.add_argument("--layer",
                default = 10,
	            help = "Number of layers. Default is 10.")

    parser.add_argument("--lr",
                default = 0.02,
	            help = "Learning rate. Default is 0.02.")

    parser.add_argument("--seed",
                        type = int,
                        default = 2021,
	                help = "Random seed for train-test split. Default is 2021.")

    parser.add_argument("--device",
                        type = str,
                        default = 'cpu',
	                help = "Cuda or CPU. Default is cpu.")

    args = parser.parse_args()

    datasets = {1:'ALOI',2:'Caltech101-7',3:'Caltech101-20',4:'MNIST',5:'MSRC-v1',6:'NUS-WIDE',7:'Youtube',8:'ORL'}
    
    dataset_name = datasets[args.dataset_id]

    # For best performance, early-stop is adopted
    # Selected numbers of epochs in our experiments
    epoch_dict = {'ALOI':55, 'Caltech101-7':50, 'Caltech101-20':45, 'MNIST': 25, 'NUS-WIDE':40, 'Youtube': 45, 'ORL': 35, 'MSRC-v1':35}

    parser.set_defaults(dataset_name = dataset_name)
    parser.set_defaults(epoch = epoch_dict[dataset_name])
    

    return parser.parse_args()

def parameter_parser_classification():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a good quality representation without grid search.
    The experimental results are obtained by setting following default hyperparameters
    """

    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--dataset-id",
                    type = int,
                default = 5)
    
    parser.add_argument("--layer",
                default = 10,
	            help = "Number of layers. Default is 10.")

    parser.add_argument("--lr",
                default = 0.05,
	            help = "Learning rate. Default is 0.05.")

    parser.add_argument("--seed",
                        type = int,
                        default = 2021,
	                help = "Random seed for train-test split. Default is 2021.")

    parser.add_argument("--ratio",
                        type = int,
                        default = 0.1,
	                help = "Ratio of superivision information. Default is 0.1.")

    parser.add_argument("--device",
                        type = str,
                        default = 'cuda',
	                help = "Cuda or CPU. Default is cpu.")

    args = parser.parse_args()

    datasets = {1:'ALOI',2:'Caltech101-7',3:'Caltech101-20',4:'MNIST',5:'MSRC-v1',6:'NUS-WIDE',7:'Youtube',8:'ORL'}
    
    dataset_name = datasets[args.dataset_id]

    # For best performance, early-stop is adopted
    # Selected numbers of epochs in our experiments
    epoch_dict = {'ALOI':10, 'Caltech101-7':15, 'Caltech101-20':10, 'MNIST': 5, 'NUS-WIDE': 10, 'Youtube': 15, 'ORL': 10, 'MSRC-v1':10}

    parser.set_defaults(dataset_name = dataset_name)
    parser.set_defaults(epoch = epoch_dict[dataset_name])
    

    return parser.parse_args()


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())

