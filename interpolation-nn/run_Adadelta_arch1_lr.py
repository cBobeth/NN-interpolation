import os
import textwrap
import pickle
from src.run_nn import Run_Model
from src.configurable_nn import Configurable_Linear_NN

#--------+---------+---------+---------+---------+---------+---------+---------+
# Set up some global variables

proj_path = '/home/bobeth/Work/digethic-AI-course/NN-interpolation'
data_path = 'data'              # original data
plot_path = 'runs/plots'        # plots from training runs
model_path = 'runs/models'      # saved trained models

data_file_name = 'fit-SM-Vcb-pmc.csv'

N_samples = 100000 # = 'All'
train_size = 0.7
valid_size = 0.2

#--------+---------+---------+---------+---------+---------+---------+---------+
# Load 'interpolate_nn.py' for common code

fn = os.path.join(proj_path, 'interpolation-nn/src', 'interpolate_nn.py')
exec(compile(source=textwrap.dedent(open(fn).read()), filename=fn, mode='exec'))

#--------+---------+---------+---------+---------+---------+---------+---------+
# Run training with specific parameters

run_lst = []
data = [X_train, Y_train, X_valid, Y_valid, X_test, Y_test]

lr_lst = [1.4, 1.0, 0.7, 0.4, 0.1, 0.07, 0.04, 0.01]

for lr in lr_lst:
    run = Run_Model(f"run_Adadelta_arch1_lr-{lr}",
                    Configurable_Linear_NN, n_in, arch_1,
                    hyper_Adadelta_MSE(lr, 2000, 16), data)
    run_lst.append(run)
    run.plot_loss_hst(y_scale='log', y_max=0.04)
    run.plot_error_hst()
    run.plot_outliers()

#--------+---------+---------+---------+---------+---------+---------+---------+
# Save run_lst

run_lst_file_name = 'Adadelta_arch1_lr.pkl'
fn = os.path.join(proj_path, model_path, run_lst_file_name)

with open(fn, 'wb') as outp:
    pickle.dump(run_lst, outp, pickle.HIGHEST_PROTOCOL)

#with open(fn, 'rb') as inp:
#    run_lst = pickle.load(inp)

print(f"\n\nFinished.\n")