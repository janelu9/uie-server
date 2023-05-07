from paddlenlp.transformers import AutoTokenizer
import paddle
import numpy as np

static_model_file="/root/.paddlenlp/taskflow/information_extraction/uie-base/static/inference.pdmodel"
static_params_file="/root/.paddlenlp/taskflow/information_extraction/uie-base/static/inference.pdiparams"

def get_bool_ids_greater_than(probs, limit=0.5, return_prob=True):
    return [[(i, p) if return_prob else (i,) for i, p in enumerate(prob) if p > limit]  for prob in probs]

def sel_max(arr):
    m=0
    for i,p in arr:
        if p>m:
            m=p
            a=i
    return a,m

def get_span(start_ids, end_ids):
    start_ids = sorted(start_ids, key=lambda x: x[0])
    end_ids = sorted(end_ids, key=lambda x: x[0])
    pre_p = 0
    sp = []
    ep = []
    result= []
    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    while start_pointer < len_start and end_pointer < len_end:
        start_id = start_ids[start_pointer]
        end_id = end_ids[end_pointer]
        if start_id[0] == end_id[0]:
            if pre_p == 0:
                sp.append(start_id)
                ep.append(end_id)
            else:
                if sp and ep:
                    result.append((sel_max(sp),sel_max(ep)))
                sp = [start_id]
                ep = [end_id]
            start_pointer += 1
            end_pointer += 1
            pre_p = 1
        elif start_id[0] <end_id[0]:
            if pre_p == 0:
                sp.append(start_id)
            else:
                if sp :
                    result.append((sel_max(sp),sel_max(ep)))
                sp = [start_id]
                ep = []
            start_pointer += 1
            pre_p = 0
        else :
            ep.append(end_id)
            end_pointer += 1
            pre_p = 1
    ep.extend(end_ids[end_pointer:])
    if sp and ep :
        result.append((sel_max(sp),sel_max(ep)))
    return result

def get_id_and_prob(span_set,bais,offset_mapping):
    sentence_id = []
    prob = []
    for start, end in span_set:
        s = start[0]-bais
        e = end[0]-bais
        if e>=s>=0:
            start_id = offset_mapping[s][0]
            end_id = offset_mapping[e][1]
            sentence_id.append((start_id, end_id))
            prob.append(start[1] * end[1])
    return sentence_id, prob

def binary_search(ori_offset,offset_mapping):
    s =0 
    e =len(offset_mapping)-1
    while s<=e:
        m = (s+e)//2
        a,b=offset_mapping[m]
        if a<=ori_offset<b:
            return m
        elif ori_offset<a:
            e=m-1
        else:
            s=m+1
    return -1

class UIEInferModel:
    def __init__(self,static_model_file,static_params_file):
        config = paddle.inference.Config(static_model_file, static_params_file)
        config.disable_gpu()
        config.enable_mkldnn()
        #config.enable_use_gpu(1000, 0)
        config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        config.delete_pass("fused_multi_transformer_encoder_pass")
        config.set_cpu_math_library_num_threads(16)
        config.switch_use_feed_fetch_ops(False)
        config.disable_glog_info()
        config.enable_memory_optim()
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handles = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]
        self.tokenizer = AutoTokenizer.from_pretrained("uie-base")

    def preprocess(self,encoded_promts,text,prompts_len,pads):
        encoded_contents = self.tokenizer(text=text,return_token_type_ids= False,return_offsets_mapping=True,)
        encoded_inputs= {'input_ids':[],'token_type_ids':[],'position_ids':[],'attention_mask':[]}
        c_len = len(encoded_contents['input_ids'][1:])
        c_type_ids = [1]*c_len
        for i,p_len in enumerate(prompts_len):
            pad = pads[i]
            encoded_inputs['input_ids'].append(encoded_promts['input_ids'][i]+encoded_contents['input_ids'][1:]+pad)
            encoded_inputs['token_type_ids'].append(encoded_promts['token_type_ids'][i]+c_type_ids+pad)
            encoded_inputs['position_ids'].append(list(range(0,p_len+c_len))+pad)
            encoded_inputs['attention_mask'].append([1]*(p_len+c_len)+pad)
        return encoded_inputs,encoded_contents['offset_mapping'][1:-1]            
    
    def predict(self,schemas,text,max_seq_len=2048,overlapping=256):
        encoded_promts = self.tokenizer(schema)
        prompts_len =[len(i) for i in  encoded_promts['input_ids']]
        max_plen = max(prompts_len)
        pads = [[0]*(max_plen-p_len ) for p_len in prompts_len]
        block_len = max_seq_len-max_plen-1
        step_len = block_len - overlapping
        assert block_len>2*overlapping
        point = 0
        stop=max(len(text)-overlapping,1)
        results=[]
        while point<stop:
            tmp_txt = text[point:point+block_len]
            encoded_inputs,offset_mapping=self.preprocess(encoded_promts,tmp_txt,prompts_len,pads)
            self.input_handles[0].copy_from_cpu(np.array(encoded_inputs['input_ids']))
            self.input_handles[1].copy_from_cpu(np.array(encoded_inputs['token_type_ids']))
            self.input_handles[2].copy_from_cpu(np.array(encoded_inputs['position_ids']))
            self.input_handles[3].copy_from_cpu(np.array(encoded_inputs['attention_mask']))
            self.predictor.run()
            start_prob = self.output_handles[0].copy_to_cpu().tolist()
            end_prob = self.output_handles[1].copy_to_cpu().tolist()
            start_ids_list=get_bool_ids_greater_than(start_prob)
            end_ids_list=get_bool_ids_greater_than(end_prob)
            result={}
            for start_ids, end_ids, bais,prompt in zip(start_ids_list,end_ids_list,prompts_len,schema):
                span_set = get_span(start_ids, end_ids)
                result[prompt]=[{'text':text[s+point:e+point],
                                 'start':s+point,
                                 'end':e+point,
                                 'probability':p} for (s,e),p in zip(*get_id_and_prob(span_set,bais,offset_mapping))]
            results.append(result)
            point+=step_len
        return self.postprocess(results,schema,step_len,overlapping)
    
    def postprocess(self,results,schema,step_len,overlapping):
        point=step_len
        result0= results[0]
        result_olp = {}
        result = {}
        for prompt in schema:
            result[prompt]=[]
            result_olp[prompt]=[]
            for item in result0[prompt]:
                if item['end']<=point:
                    result[prompt].append(item)
                else:
                    result_olp[prompt].append(item)
        for block in results[1:]:
            result_olp_tail={}
            for prompt in schema:
                for item in block[prompt]:
                    if item['start']>=point+overlapping and item['end']<=point+step_len:
                        result[prompt].append(item)
                    elif item['start']<point+overlapping :
                        for olp in result_olp[prompt]:
                            if not (item['start']>olp['end'] or item['end']<olp['start']):
                                item['start'] =min(item['start'],olp['start'])
                                item['end'] =max(item['end'],olp['end'])
                                item['probability'] =max(item['probability'],olp['probability'])
                                item['text']=text[item['start']:item['end']]
                                result_olp[prompt].remove(olp)
                        if item['end']<=point+step_len:
                            result[prompt].append(item)
                        else:
                            result_olp_tail[prompt].append(item)      
                    else:
                        result_olp_tail[prompt].append(item)
                result[prompt].extend(result_olp[prompt])
            result_olp=result_olp_tail
            point+=step_len
        for prompt in result_olp:  
            result[prompt].extend(result_olp[prompt]) 
        return result
    
    def __call__(self,schemas,text,max_seq_len=2048,overlapping=256,schema_size = 8):
        result={}
        for i in range(0,len(schemas),schema_size):
            result.update(self.predict(schemas[i:i+schema_size],text,max_seq_len,overlapping))
        return result