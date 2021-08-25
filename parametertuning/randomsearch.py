from numpy import random
import numpy as np

class RandomSearchJob(object):
    def __init__(self, data_name, search_space: dict, mode='train'):
        self.data_name = data_name
        self.search_space = search_space
        self.mode = mode

    def single_task_trial(self, random_seed):
        """
        according to the parameter order in run.sh
        """
        parameter_list = [[] for _ in range(20 + 1)]
        parameter_dict = {}
        parameter_list[1] = self.mode
        parameter_dict['mode'] = str(self.mode)
        parameter_list[2] = self.data_name
        parameter_dict['data'] = str(self.data_name)
        # ++++++++++++++++++++++++++++++++++++++++++++
        num_layers = self.rand_search_parameter(self.search_space['num_layers'])
        parameter_list[4] = num_layers
        parameter_dict['layers'] = str(num_layers)

        hidden_dim = self.rand_search_parameter(self.search_space['hidden_dim'])
        parameter_list[5] = hidden_dim
        parameter_dict['H'] = str(hidden_dim)

        project_dim = self.rand_search_parameter(self.search_space['project_dim'])
        parameter_list[6] = project_dim
        parameter_dict['P'] = str(project_dim)

        num_heads = self.rand_search_parameter(self.search_space['num_heads'])
        parameter_list[7] = num_heads
        parameter_dict['HDs'] = str(num_heads)

        hop_num = self.rand_search_parameter(self.search_space['hop_num'])
        parameter_list[8] = hop_num
        parameter_dict['Hop'] = str(hop_num)

        lr = self.rand_search_parameter(self.search_space['learning_rate'])
        lr = round(lr, 5)
        parameter_list[9] = lr
        parameter_dict['lr'] = str(lr)

        fea_drop = self.rand_search_parameter(self.search_space['fea_drop'])
        parameter_list[10] = round(fea_drop, 4)
        parameter_dict['fdr'] = str(round(fea_drop, 4))

        att_drop = self.rand_search_parameter(self.search_space['att_drop'])
        parameter_list[11] = round(att_drop, 4)
        parameter_dict['adr'] = str(round(att_drop, 4))

        neg_slope = self.rand_search_parameter(self.search_space['negative_slope'])
        neg_slope = round(neg_slope, 3)
        parameter_list[12] = neg_slope
        parameter_dict['nl'] = str(neg_slope)

        adam_decacy = self.rand_search_parameter(self.search_space['adam_weight_decay'])
        parameter_list[13] = adam_decacy
        parameter_dict['dec'] = str(adam_decacy)

        epochs = self.rand_search_parameter(self.search_space['epochs'])
        parameter_list[14] = epochs
        parameter_dict['ep'] = str(epochs)

        alpha = self.rand_search_parameter(self.search_space['alpha'])
        parameter_list[15] = round(alpha, 2)
        parameter_dict['alpha'] = str(round(alpha, 2))

        topk = self.rand_search_parameter(self.search_space['top_k'])
        parameter_list[16] = topk
        parameter_dict['topk'] = str(topk)

        topk_type = self.rand_search_parameter(self.search_space['topk_type'])
        parameter_list[17] = topk_type
        parameter_dict['kt'] = str(topk_type)

        edge_drop = self.rand_search_parameter(self.search_space['edge_drop'])
        edge_drop = round(edge_drop, 3)
        parameter_list[18] = edge_drop
        parameter_dict['ed'] = str(edge_drop)

        self_loop = self.rand_search_parameter(self.search_space['self_loop'])
        parameter_list[19] = self_loop
        parameter_dict['sl'] = str(self_loop)

        parameter_list[20] = random_seed
        parameter_dict['seed'] = str(random_seed)

        task_id = '_'.join([k+'_' + v for k, v in parameter_dict.items()])
        parameter_list[3] = task_id
        parameter_id = ' '.join([str(para) for idx, para in enumerate(parameter_list) if idx > 0])
        return task_id, parameter_id

    def rand_search_parameter(self, space: dict):
        para_type = space['type']
        if para_type == 'fixed':
            return space['value']

        if para_type == 'choice':
            candidates = space['values']
            value = random.choice(candidates, 1)[0]
            return value

        if para_type == 'range':
            log_scale = space.get('log_scale', False)
            low, high = space['bounds']
            if log_scale:
                value = random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
                value = np.exp(value)
            else:
                value = random.uniform(low=low, high=high,size=1)[0]
            return value
        else:
            raise ValueError('Training batch mode %s not supported' % para_type)