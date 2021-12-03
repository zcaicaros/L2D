import numpy
import numpy as np
import matplotlib.pyplot as plt
import re

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
show = True
save = False
save_file_type = '.pdf'
# problem params
n_j = 15
n_m = 15
l = 1
h = 99
stride = 50
datatype = 'vali'  # 'vali', 'log'


f = open('./{}_{}_{}_{}_{}.txt'.format(datatype, n_j, n_m, l, h), 'r').readline()
if datatype == 'vali':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f)[1::2]])[:]
    idx = np.arange(obj.shape[0])
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('MakeSpan', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('message-passing_time', save_file_type))
    if show:
        plt.show()
elif datatype == 'log':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f)[1::2]])[:].reshape(-1, stride).mean(axis=-1)
    idx = np.arange(obj.shape[0])
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('MakeSpan', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('message-passing_time', save_file_type))
    if show:
        plt.show()
else:
    print('Wrong datatype.')



