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

N_samples = 'All'
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

bs_lst = [512, 256, 128, 64, 32, 16, 8, 4]

for bs in bs_lst:
    run = Run_Model(f"run_SGD-mom_arch4_bs-B-{bs}",
                    Configurable_Linear_NN, n_in, arch_4,
                    hyper_SGD_MSE(0.07, 2000, bs, 0.2), data)
    run_lst.append(run)
    run.plot_loss_hst(y_scale='log', y_max=0.04)
    run.plot_error_hst()
    run.plot_outliers()

#--------+---------+---------+---------+---------+---------+---------+---------+
# Save run_lst

run_lst_file_name = 'SGD-mom_arch4_bs.pkl'
fn = os.path.join(proj_path, model_path, run_lst_file_name)

with open(fn, 'wb') as outp:
    pickle.dump(run_lst, outp, pickle.HIGHEST_PROTOCOL)

#with open(fn, 'rb') as inp:
#    run_lst = pickle.load(inp)

print(f"\n\nFinished.\n")