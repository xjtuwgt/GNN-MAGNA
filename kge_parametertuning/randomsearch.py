from numpy import random
import numpy as np

class KGERandomSearchJob(object):
    def __init__(self, data_name, search_space: dict, mode='train', graph_on=0,
                 trans_on=0, mask_on=0):
        self.data_name = data_name
        self.graph_on = graph_on
        self.mask_on = mask_on
        self.search_space = search_space
        self.mode = mode
        self.trans_on = trans_on
        self.gpu_device = 'nan'

    def single_task_trial(self, random_seed):
        """
        according to the parameter order in run.sh
        """
        parameter_list = [[] for _ in range(38 + 1)]
        parameter_dict = {}
        parameter_list[1] = self.mode
        parameter_dict['ME'] = str(self.mode)
        model_name = self.rand_search_parameter(self.search_space['model_name'])
        parameter_list[2] = model_name
        parameter_dict['ML'] = str(model_name)
        parameter_list[3] = self.data_name
        parameter_dict['DA'] = str(self.data_name)
        parameter_list[4] = self.gpu_device
        #++++++++++++++++++++++++++++++++++++++++++++
        batch_size = self.rand_search_parameter(self.search_space['batch_size'])
        parameter_list[6] = batch_size
        parameter_dict['bs'] = str(batch_size)

        hidden_dim = self.rand_search_parameter(self.search_space['hidden_dim'])
        parameter_list[7] = hidden_dim
        parameter_dict['H'] = str(hidden_dim)

        ent_dim = self.rand_search_parameter(self.search_space['ent_embed_dim'])
        parameter_list[8] = ent_dim
        parameter_dict['E'] = str(ent_dim)

        rel_dim = self.rand_search_parameter(self.search_space['rel_embed_dim'])
        parameter_list[9] = rel_dim
        parameter_dict['R'] = str(rel_dim)

        emb_dim = self.rand_search_parameter(self.search_space['embed_dim'])
        parameter_list[10] = emb_dim
        parameter_dict['D'] = str(emb_dim)

        gamma = self.rand_search_parameter(self.search_space['gamma'])
        parameter_list[11] = round(gamma, 4)
        parameter_dict['g'] = str(round(gamma, 4))

        alpha = self.rand_search_parameter(self.search_space['alpha'])
        parameter_list[12] = round(alpha, 2)
        parameter_dict['a'] = str(round(alpha, 2))

        lr = self.rand_search_parameter(self.search_space['learning_rate'])
        parameter_list[13] = round(lr, 5)
        parameter_dict['lr'] = str(round(lr, 5))

        max_steps = self.rand_search_parameter(self.search_space['max_steps'])
        parameter_list[14] = max_steps
        parameter_dict['max'] = str(max_steps)

        parameter_list[15] = 16 #Test batch size

        att_drop = self.rand_search_parameter(self.search_space['att_drop'])
        parameter_list[16] = round(att_drop, 4)
        parameter_dict['adr'] = str(round(att_drop, 4))

        fea_drop = self.rand_search_parameter(self.search_space['fea_drop'])
        parameter_list[17] = round(fea_drop, 4)
        parameter_dict['fdr'] = str(round(fea_drop, 4))

        inp_drop = self.rand_search_parameter(self.search_space['inp_drop'])
        parameter_list[18] = round(inp_drop, 4)
        parameter_dict['idr'] = str(round(inp_drop, 4))

        topk = self.rand_search_parameter(self.search_space['top_k'])
        parameter_list[19] = topk
        parameter_dict['k'] = str(topk)

        hops = self.rand_search_parameter(self.search_space['hops'])
        parameter_list[20] = hops
        parameter_dict['ho'] = str(hops)

        layers = self.rand_search_parameter(self.search_space['layers'])
        parameter_list[21] = layers
        parameter_dict['ler'] = str(layers)

        parameter_list[22] = self.trans_on
        parameter_list[23] = self.graph_on
        parameter_list[24] = self.mask_on
        loss_type = self.rand_search_parameter(self.search_space['loss_type'])
        parameter_list[25] = loss_type
        neg_on = self.rand_search_parameter(self.search_space['negative_on'])
        parameter_list[26] = neg_on
        project_on = self.rand_search_parameter(self.search_space['project_on'])
        parameter_list[27] = project_on
        parameter_dict['TGMLNP'] = str(self.trans_on) + str(self.graph_on) + str(self.mask_on) + str(loss_type) + str(neg_on) + str(project_on)

        num_heads = self.rand_search_parameter(self.search_space['num_heads'])
        parameter_list[28] = num_heads
        parameter_dict['Hs'] = str(num_heads)

        adam_decacy = self.rand_search_parameter(self.search_space['adam_weight_decay'])
        parameter_list[29] = adam_decacy
        parameter_dict['dec'] = str(adam_decacy)
        conv_embed_shape1 = self.rand_search_parameter(self.search_space['conv_embed_shape1']) #For TranConvE, this is equal to the channels
        parameter_dict['cv'] = str(conv_embed_shape1)
        parameter_list[30] = conv_embed_shape1

        conv_filter_size = self.rand_search_parameter(self.search_space['conv_filter_size'])
        parameter_dict['cf'] = str(conv_filter_size)
        parameter_list[31] = conv_filter_size

        conv_channels = self.rand_search_parameter(self.search_space['conv_channels'])
        parameter_dict['cc'] = str(conv_channels)
        parameter_list[32] = conv_channels

        topk_type = self.rand_search_parameter(self.search_space['topk_type'])
        parameter_dict['tt'] = str(topk_type)
        parameter_list[33] = topk_type

        feed_forward = self.rand_search_parameter(self.search_space['feed_forward'])
        parameter_dict['ff'] = str(feed_forward)
        parameter_list[34] = feed_forward

        edge_drop = self.rand_search_parameter(self.search_space['edge_drop'])
        parameter_dict['ed'] = str(round(edge_drop, 2))
        parameter_list[35] = edge_drop

        reg = self.rand_search_parameter(self.search_space['regularization'])
        parameter_dict['r'] = str(round(reg, 2))
        parameter_list[36] = reg

        reg_type = self.rand_search_parameter(self.search_space['reg_type'])
        parameter_dict['rt'] = str(reg_type)
        parameter_list[37] = reg_type

        parameter_list[38] = random_seed
        parameter_dict['s'] = str(random_seed)

        task_id = ''.join([k + v for k, v in parameter_dict.items()])
        parameter_list[5] = task_id
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
                value = random.uniform(low=np.log(low), high=np.log(high),size=1)[0]
                value = np.exp(value)
            else:
                value = random.uniform(low=low, high=high,size=1)[0]
            return value
        else:
            raise ValueError('Training batch mode %s not supported' % para_type)