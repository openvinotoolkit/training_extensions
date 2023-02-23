import numpy as np
import os
import random
import argparse
import torch
import operator
import math
from torch.backends import cudnn
from torchvision import  transforms
import onnx
import onnxruntime
from openvino.inference_engine import IECore
from src.utils.network import Encoder, load_checkpoint
from src.utils.vectorHandle import  precision, re_classes, discounted_cumulative_gain

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class RetrievalInference():
    def __init__(self, model, gallery_path, query_path, zSize,  checkpoint, device):
        self.device = device
        self.model = model.to(self.device)
        self.checkpoint = checkpoint
        load_checkpoint(self.model, self.checkpoint)
        self.gallery_path = gallery_path
        self.query_path = query_path
        self.zSize = zSize

    def test_onnx(self, sample_image_path, onnx_checkpoint):
        onnx_model = onnx.load(onnx_checkpoint)
        onnx.checker.check_model(onnx_model)
        # The validity of the ONNX graph is verified by checking the model’s version,
        # the graph’s structure, as well as the nodes and their inputs and outputs.
        ort_session = onnxruntime.InferenceSession(onnx_checkpoint)
        to_tensor = transforms.ToTensor()
        sample_image = np.load(sample_image_path)
        ort_inputs = {ort_session.get_inputs()[0].name: sample_image}
        ort_inputs['input'] = np.reshape(ort_inputs['input'].astype('float32'), [1,1,28,28])
        _ = ort_session.run(None, ort_inputs, None)
        sample_image = to_tensor(sample_image)
        sample_image = torch.reshape(sample_image, [1,1,28,28]).cuda()
        _ = self.model(sample_image)

    def load_inference_model(self, run_type, onnx_checkpoint):
        if run_type == 'pytorch':
            model = self.model
        elif run_type == 'onnx':
            model = onnxruntime.InferenceSession(onnx_checkpoint)
        else:
            ie = IECore()
            model_xml = os.path.splitext(onnx_checkpoint)[0] + ".xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
            model_temp = ie.read_network(model_xml, model_bin)
            model = ie.load_network(network=model_temp, device_name='CPU')
        return model


    def gallery_building(self, model, run_type ='pytroch'):
        gallery = {}
        to_tensor = transforms.ToTensor()
        gallery_count = len(os.listdir(self.gallery_path))
        query_count = len(os.listdir(self.query_path))
        print(f"Number of gallery images:{gallery_count}")
        print(f"Number of gallery images:{query_count}")
        gNimage  = random.sample(os.listdir(self.gallery_path), gallery_count)
        _ = random.sample(os.listdir(self.query_path), query_count)
        print("\n\n Building Gallery .... \n")
        for img in gNimage:
            np_img = np.load(os.path.join(self.gallery_path, img))
            im = np.resize(np_img,(28, 28))
            numpy_image = np.array(im)
            if len(numpy_image.shape) < 3:
                numpy_image = np.stack((numpy_image,)*1, axis=-1)
            numpy_image1 = numpy_image.transpose((2, 0, 1))
            numpy_image = np.array([numpy_image1])
            torch_image = torch.from_numpy(numpy_image)
            torch_image = torch_image.type('torch.cuda.FloatTensor')
            if run_type == 'pytorch':
                hashcode, _ = model(torch_image)

            elif run_type == 'onnx':
                ort_inputs = {model.get_inputs()[0].name: numpy_image1}
                ort_inputs['input'] = np.reshape(ort_inputs['input'].astype('float32'), [1,1,28,28])
                out = model.run(None, ort_inputs, None)
                hashcode = np.array(out[0])
                hashcode = to_tensor(hashcode).to(self.device)
            else:
                out = model.infer(inputs={'input':numpy_image})['output']
                hashcode = to_tensor(out).to(self.device)

            gallery[img] = hashcode
            del torch_image
        print("\n Building Complete. \n")
        return gallery


    def distance(self, q_name, gallery):
        query_image = os.path.join(self.query_path, q_name)
        np_im_q =np.load(query_image)
        im_q = np.resize(np_im_q ,(28, 28))
        numpy_image_q = np.array(im_q)
        if len(numpy_image_q.shape) < 3:
            numpy_image_q = np.stack((numpy_image_q,)*1, axis=-1)
        numpy_image_q = (numpy_image_q.transpose((2, 0, 1)))
        numpy_image_q = np.array([numpy_image_q])
        torch_image_q = torch.from_numpy(numpy_image_q)
        torch_image_q = torch_image_q.type("torch.cuda.FloatTensor")
        h_q, _ = self.model(torch_image_q)
        dist = {}
        for key in gallery.keys():
            h1 = gallery[key]
            h1norm = torch.div(h1, torch.norm(h1, p=2))
            h2norm = torch.div(h_q, torch.norm(h_q, p=2))
            dist[key] = torch.pow(torch.norm(h1norm - h2norm, p=2), 2)*self.zSize/4
        return dist

    def mean_average_precision(self, gallery):
        qNimage  = random.sample(os.listdir(self.query_path), int(len(os.listdir(self.query_path))))
        count_map= 0
        q_prec = 0
        q_prec_100 = 0
        q_prec_1000 = 0
        q_prec_10000 = 0

        for q_name in qNimage:
            q_class = q_name.split(".")[0].split("_")[0]
            count_map = count_map+1
            dist = self.distance(q_name,  gallery)
            sorted_pool = sorted(dist.items(), key=operator.itemgetter(1))[0:10]
            ret_classes = [sorted_pool[i][0].split(".")[0].split("_")[0]
                                    for i in range(len(sorted_pool))]
            q_prec += precision(q_class, ret_classes)

            sorted_pool_100 = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
            ret_classes_100 = [sorted_pool_100[i][0].split(".")[0].split("_")[0]
                                for i in range(len(sorted_pool_100))]
            q_prec_100 += precision(q_class, ret_classes_100)
            sorted_pool_1000 = sorted(dist.items(), key=operator.itemgetter(1))[0:1000]
            ret_classes_1000 = [sorted_pool_1000[i][0].split(".")[0].split("_")[0]
                                for i in range(len(sorted_pool_1000))]
            q_prec_1000 += precision(q_class, ret_classes_1000)
            sorted_pool_10000 = sorted(dist.items(), key=operator.itemgetter(1))[0:10000]
            ret_classes_10000 = [sorted_pool_10000[i][0].split(".")[0].split("_")[0]
                                    for i in range(len(sorted_pool_10000))]
            q_prec_10000 += precision(q_class, ret_classes_10000)
            print(count_map)
        print("Model" + " ::  mAP@10 :", q_prec/len((qNimage)))
        print("Model" + " ::  mAP@100 :", q_prec_100/len((qNimage)))
        print("Model" + " ::  mAP@1000 :", q_prec_1000/len((qNimage)))

        return q_prec/len((qNimage)), q_prec_100/len((qNimage)), q_prec_1000/len((qNimage)), q_prec_10000/len((qNimage))
    def nornalized_discounted_cumulative_gain(self, gallery):
        qNimage  = random.sample(os.listdir(self.query_path), int(len(os.listdir(self.query_path))))
        count_ndcg= 0
        ndcg_im_10=[]
        ndcg_im_100 = []
        ndcg_im_1000 = []
        ndcg_im_10000 = []

        for q_name in qNimage:
            count_ndcg = count_ndcg+1
            dist = self.distance(q_name, gallery)
            sorted_pool_10 = sorted(dist.items(), key=operator.itemgetter(1))[0:10]
            set1_10,set2_10 = re_classes(sorted_pool_10,q_name)
            dcg_10 = discounted_cumulative_gain(set1_10)
            idcg_10 = discounted_cumulative_gain(set2_10)
            if dcg_10 and idcg_10:
                ndcg_im_10.append(dcg_10 / idcg_10)
            #nDCG for top 100 retrieval
            sorted_pool_100 = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
            set1_100,set2_100 = re_classes(sorted_pool_100,q_name)
            dcg_100 = discounted_cumulative_gain(set1_100)
            idcg_100 = discounted_cumulative_gain(set2_100)
            if dcg_100 and idcg_100:
                ndcg_im_100.append(dcg_100 / idcg_100)

            # nDCG for top 100 retrieval
            sorted_pool_1000 = sorted(dist.items(), key=operator.itemgetter(1))[0:1000]
            set1_1000,set2_1000 = re_classes(sorted_pool_1000,q_name)
            dcg_1000 = discounted_cumulative_gain(set1_1000)
            idcg_1000 = discounted_cumulative_gain(set2_1000)
            if dcg_1000 and idcg_1000:
                ndcg_im_1000.append(dcg_1000 / idcg_1000)
            #nDCG for top 1000 retrieval
            sorted_pool_10000 = sorted(dist.items(), key=operator.itemgetter(1))[0:10000]
            set1_10000,set2_10000 = re_classes(sorted_pool_10000,q_name)
            dcg_10000 = discounted_cumulative_gain(set1_10000)
            idcg_10000 = discounted_cumulative_gain(set2_10000)
            if dcg_10000 and idcg_10000:
                ndcg_im_10000.append(dcg_10000 / idcg_10000)
            print(count_ndcg)
        ndcg_10 = [x for x in ndcg_im_10 if math.isnan(x) is False] # to avoid empty list
        ndcg_10 = sum(ndcg_10) / len(ndcg_im_10)

        ndcg_100 = [x for x in ndcg_im_100 if math.isnan(x) is False]
        ndcg_100 = sum(ndcg_100) / len(ndcg_im_100)
        ndcg_1000 = [x for x in ndcg_im_1000 if math.isnan(x) is False]
        ndcg_1000 = sum(ndcg_1000) / len(ndcg_im_1000)

        ndcg_10000 = [x for x in ndcg_im_10000 if math.isnan(x) is False]
        ndcg_10000 = sum(ndcg_10000) / len(ndcg_im_10000)
        print('NDCG@10', ndcg_10)
        print('NDCG@100', ndcg_100)
        print('NDCG@1000', ndcg_1000)
        return ndcg_10, ndcg_100, ndcg_1000, ndcg_10000

    def validate_models(self, run_type, onnx_checkpoint =''):
        cudnn.benchmark = True
        model = self.load_inference_model(run_type, onnx_checkpoint)
        gallery = self.gallery_building(model, run_type)
        print("\n Start calculating mean average precision: \n")
        map_10, map_100, map_1000, map_10000 = self.mean_average_precision( gallery)
        print('mAP completed')
        print("\n Start calculating normalized discounted cumulative:")
        ndcg_10, ndcg_100, ndcg_1000, ndcg_10000 = self.nornalized_discounted_cumulative_gain( gallery)
        print('nDCG completed')


        return map_10, map_100, map_1000, map_10000, ndcg_10, ndcg_100, ndcg_1000, ndcg_10000

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main(args):
    checkpoint = args.checkpoint
    zSize = args.zSize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available
    savepath = args.modelpath
    current_dir =  os.path.abspath(os.path.dirname(__file__))

    # Data Path
    gallery_path = current_dir + os.path.join(args.dpath, 'gallery')
    query_path = current_dir + os.path.join(args.dpath, 'query')


    #Construct Model
    model = Encoder(zSize)
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    model_path = current_dir + os.path.join(savepath, 'encoder-100.pkl')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.cuda()

    retrieval = RetrievalInference(model, gallery_path, query_path, zSize, checkpoint, device)
    _,_,_,_,_, _,_,_, = retrieval.validate_models(run_type = 'pytorch')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        required=False,
        help="start training from a checkpoint model weight",
        default= None,
        type = str)
    parser.add_argument("--dpath",
        required=False,
        default = '/utils/dataset',
        help="Path to folder containing all data",
        type =str)
    parser.add_argument("--modelpath",
        required=False,
        default = '/utils/model_weights/',
        help="Path to folder from which models would be loaded",
        type =str)
    parser.add_argument("--zSize",
        required=False,
        help="hash code length for the model",
        default=48,
        type=float)
    custom_args = parser.parse_args()

    main(custom_args)
