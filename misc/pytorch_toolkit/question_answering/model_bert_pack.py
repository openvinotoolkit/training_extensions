"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import logging
import torch


from transformers.modeling_bert import BertEmbeddings
from transformers.modeling_bert import BertEncoder
from transformers.modeling_bert import BertSelfAttention
from transformers.modeling_bert import BertSelfOutput
from transformers.modeling_bert import BertIntermediate
from transformers.modeling_bert import BertOutput
from transformers.modeling_bert import BertPooler
from transformers.modeling_bert import BertModel
from transformers.activations import ACT2FN

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} model_bert_pack'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

def get_modules(net, net_name=None):
    class Desc():
        def __init__(self, name_full, module, name, module_parent):
            self.name_full = name_full
            self.module = module
            self.name = name
            self.module_parent = module_parent

    modules = []
    for n, m in net._modules.items():
        m_name = n if net_name == None else (net_name + "." + n)
        modules.append(Desc(
            name_full=m_name,
            module=m,
            name=n,
            module_parent=net
        ))
        modules += get_modules(m, m_name)
    return modules


class EmbeddingPacked(torch.nn.Module):
    def __init__(self, embedding, pack_cfg):
        super().__init__()
        hidden_size = int(pack_cfg.get('hidden_size', embedding.embedding_dim))
        self.embedding = embedding
        self.linear = torch.nn.Linear(embedding.embedding_dim, hidden_size)

    def merge_linear_(self):
        weight_old = self.embedding.weight.data
        with torch.no_grad():
            weight_new = self.linear(weight_old)
            self.embedding.weight.data = weight_new.data.detach()
            self.embedding.embedding_dim = self.embedding.weight.shape[1]
        del self.linear


    def forward(self, *args, **kwargs):
        output = self.embedding(*args, **kwargs)
        if hasattr(self,'linear'):
            output = self.linear(output)
        return output


class BertIntermediatePacked(torch.nn.Module):
    def __init__(self, m, pack_cfg):
        super().__init__()
        hidden_size = int(pack_cfg.get('hidden_size', m.dense.in_features))
        iter_num = int(pack_cfg.get('ff_iter_num', 2))
        self.denses = torch.nn.ModuleList()
        for i in range(iter_num-1):
            self.denses.append( torch.nn.Linear(hidden_size, hidden_size) )

        nn_func = {'orig': m.intermediate_act_fn}
        nn_func.update(ACT2FN)
        nn_func_name = pack_cfg.get('hidden_act', 'orig')
        assert nn_func_name in nn_func, "{} hidden_act is not supported. The supported list is {}".format(nn_func_name, list(nn_func.keys()))
        self.intermediate_act_fn = nn_func[nn_func_name]

    def forward(self, hidden_states):
        for dense in self.denses:
            hidden_states = dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class BertEncoderPacked(BertEncoder):
    def __init__(self, config, m, pack_cfg):
        super().__init__(config)

        self.pack_emb =  pack_cfg['pack_emb']
        num_hidden_layers_old = len(m.layer)
        num_hidden_layers = int(pack_cfg.get('num_hidden_layers', num_hidden_layers_old))
        if num_hidden_layers != num_hidden_layers_old:
            logger.info("PACK reduce number of layers from {} to {}".format(num_hidden_layers_old, num_hidden_layers))
            layers = []
            for i in range(num_hidden_layers):
                j = num_hidden_layers_old * (i + 1) // num_hidden_layers - 1
                layers.append( m.layer[j] )
            self.layer = torch.nn.ModuleList(layers)
        else:
            self.layer = m.layer

        hidden_size_old = config.hidden_size
        hidden_size = int(pack_cfg.get('hidden_size', hidden_size_old))
        #pack high level encoder by reducing hidden size
        if hidden_size != hidden_size_old:
            logger.info("PACK add linear transforms to map {} hidden size to {}".format(hidden_size_old, hidden_size))


            if self.pack_emb:
                #input embeddings already packed use input_transform to unpuck it for loss calculation
                self.input_transform = torch.nn.Linear(hidden_size, hidden_size_old )
            else:
                self.input_transform = torch.nn.Linear(hidden_size_old, hidden_size)

            ml =  [torch.nn.Linear(hidden_size, hidden_size_old) for _ in self.layer]
            self.output_transforms = torch.nn.ModuleList(ml)

    def forward(self,hidden_states, **kwargs):

        if hasattr(self,'input_transform'):
            #preapre inputs
            if self.pack_emb:
                hidden_states_out = self.input_transform(hidden_states)
            else:
                hidden_states_out = hidden_states
                hidden_states = self.input_transform(hidden_states)

        #forward base encoder
        outputs = super().forward(hidden_states, **kwargs)

        if hasattr(self,'output_transforms'):
            #preapre outputs
            if hasattr(self,'outputs_merged') and self.outputs_merged:
                outputs_new = (outputs[0],)
            else:
                output_encoder = self.output_transforms[-1](outputs[0])
                outputs_new = (output_encoder,)

            outputs = outputs[1:]

            output_hidden_states = False
            if hasattr(self,'output_hidden_states'):
                #old transformer version has output_hidden_states filed
                output_hidden_states = self.output_hidden_states
            if 'output_hidden_states' in kwargs:
                #newer transformers get this flag as input arg
                output_hidden_states = kwargs['output_hidden_states']
            if output_hidden_states:
                output_hidden_states = outputs[0]
                t = tuple(m(o) for m,o in zip(self.output_transforms,output_hidden_states[1:]))
                output_hidden_states_new = (hidden_states_out, ) + t
                outputs_new = outputs_new + (output_hidden_states_new,)
                outputs = outputs[1:]

            outputs = outputs_new + outputs

        #add the rest outputs
        return outputs



def BertPacked(BertBasedClass):
    class BertBasedClassPacked(BertBasedClass):
        def __init__(self, config):
            super().__init__(config)
            if hasattr(config,'pack_cfg'):
                self.pack_(config.pack_cfg)

        def get_bert(self):
            berts = list(filter(lambda x: isinstance(x, BertModel), self.modules()))
            assert len(berts) == 1
            return berts[0]

        def pack_(self, pack_cfg):
            device = next(self.parameters()).device

            hidden_size_old = self.config.hidden_size
            hidden_size = int(pack_cfg.get('hidden_size', hidden_size_old))

            #add new params to read old configurations without this param
            if 'pack_emb' not in pack_cfg:
                pack_cfg['pack_emb'] = False
            pack_cfg['base_class_name'] = BertBasedClass.__name__

            self.config.pack_cfg = pack_cfg

            modules = get_modules(self)

            #looking for bert.encoder
            #self.bert.encoder = BertEncoderPacked(self.config, self.bert.encoder, pack_cfg).to(device)
            for m_desc in filter(lambda x:isinstance(x.module, BertEncoder), modules):
                # pack high level encoder by reducing number of layers and adding transformation for hidden states
                module_packed = BertEncoderPacked(self.config, m_desc.module, pack_cfg).to(device)
                m_desc.module_parent._modules[m_desc.name] = module_packed
                self.config.num_hidden_layers = len(module_packed.layer)

            #get new list of module due to replacment on the previous stage
            modules = get_modules(self)
            params_packed = []
            #pack many internal modules
            for m_desc in modules:
                m = m_desc.module

                #change SelfAttention
                if isinstance(m, BertSelfAttention):
                    if hidden_size != hidden_size_old:
                        m.query = torch.nn.Linear(hidden_size, m.all_head_size).to(device)
                        m.key = torch.nn.Linear(hidden_size, m.all_head_size).to(device)
                        m.value = torch.nn.Linear(hidden_size, m.all_head_size).to(device)
                        logger.info("PACK ATT({}) packed by reducing hidden size {}->{}".format(m_desc.name_full,hidden_size_old,hidden_size ))
                if isinstance(m, BertSelfOutput):
                    if hidden_size != hidden_size_old:
                        input_size = m.dense.in_features
                        m.dense = torch.nn.Linear(input_size, hidden_size).to(device)
                        m.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=m.LayerNorm.eps).to(device)
                        logger.info("PACK ATT({}) packed by reducing hidden size {}->{}".format(m_desc.name_full, hidden_size_old, hidden_size))

                #change FF block
                if isinstance(m, BertIntermediate):
                    m_packed = BertIntermediatePacked(m, pack_cfg)
                    m_desc.module_parent._modules[m_desc.name] = m_packed.to(device)
                    logger.info("PACK FF({}) packed by replacing by new block".format(m_desc.name_full ))
                if isinstance(m, BertOutput):
                    m.dense = torch.nn.Linear(hidden_size, hidden_size).to(device)
                    m.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=m.LayerNorm.eps).to(device)
                    logger.info("PACK FF({}) packed by replacing by new block".format(m_desc.name_full))

                #change embeddings block
                if pack_cfg['pack_emb'] and isinstance(m, BertEmbeddings):
                    m.word_embeddings = EmbeddingPacked(m.word_embeddings, pack_cfg).to(device)
                    m.position_embeddings = EmbeddingPacked(m.position_embeddings, pack_cfg).to(device)
                    m.token_type_embeddings = EmbeddingPacked(m.token_type_embeddings, pack_cfg).to(device)
                    m.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=m.LayerNorm.eps).to(device)

                    params_packed += list(m.word_embeddings.linear.parameters())
                    params_packed += list(m.position_embeddings.linear.parameters())
                    params_packed += list(m.token_type_embeddings.linear.parameters())
                    params_packed += list(m.LayerNorm.parameters())

                    logger.info("PACK EMB({}) packed by reducing hidden size {}->{}".format(m_desc.name_full, hidden_size_old,hidden_size))


            #add all bert encoder parameters
            for m in filter(lambda x:isinstance(x, BertEncoderPacked), self.modules()):
                params_packed += list(m.parameters())

            return params_packed

        def merge_(self):
            # merge some operations
            for m in self.modules():
                if isinstance(m, EmbeddingPacked):
                    m.merge_linear_()


            bert_encoders = list(filter(lambda x:isinstance(x, BertEncoderPacked), self.modules()))
            assert len(bert_encoders) == 1
            bert_encoder = bert_encoders[0]

            l1 = bert_encoder.output_transforms[-1]

            #merge final transform into qa_outputs regressors
            if hasattr(self,'qa_outputs'):
                l2 = self.qa_outputs
                l21 = torch.nn.Linear(l1.weight.shape[1], l2.weight.shape[0])
                with torch.no_grad():
                    w21 = torch.matmul(l2.weight.data,l1.weight.data)
                    b21 = torch.matmul(l2.weight.data,l1.bias.data)
                    l21.weight.data = w21
                    l21.bias.data =b21
                self.qa_outputs = l21

            #merge final state transformation into pooler
            for m in filter(lambda x:isinstance(x, BertPooler),self.modules()):
                l2 = m.dense
                l21 = torch.nn.Linear(l1.weight.shape[1], l2.weight.shape[0])
                with torch.no_grad():
                    w21 = torch.matmul(l2.weight.data,l1.weight.data)
                    b21 = torch.matmul(l2.weight.data,l1.bias.data)
                    l21.weight.data = w21
                    l21.bias.data =b21
                m.dense = l21

            bert_encoder.outputs_merged = True

    return BertBasedClassPacked

