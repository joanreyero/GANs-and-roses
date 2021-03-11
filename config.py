from collections import namedtuple

epoch = 'latest'
epoch_count = 1
n_epochs = 100
n_epochs_decay = 100
print_freq = 100
display_freq = 400
update_html_freq = 1000 
display_id = 1
save_latest_freq = 5000
save_by_iter = False
batch_size = 1


CG_config = namedtuple(
    'CG_config', [
        'lambda_A', 
        'lambda_B', 
        'lambda_identity', 
        'pool_size', 
        'lr', 
        'beta1'],  
    defaults= [10.0, 10.0, 0.5, 50, 0.0002, 0.5])


