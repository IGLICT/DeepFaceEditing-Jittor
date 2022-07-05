import jittor as jt
from jittor import Module
from jittor import nn
import networks
import numpy as np
from numpy.linalg import solve

class AE_Model(nn.Module):
    def name(self):
        return 'AE_Model'
    
    def initialize(self, key):
        ##### define networks        
        # Generator network       

        #The axis of x,y; the size of each part
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}

        self.drawing_encoder_part = networks.DrawingEncoder(image_size=self.part[key][2], input_nc=1)
        self.drawing_decoder_part = networks.DrawingDecoder(image_size=self.part[key][2], output_nc=32)

        #print("load the weight of " + key)
        self.drawing_encoder_part.load('./checkpoints/Drawing/sketch_encoder_' + key + '.pkl')
        self.drawing_decoder_part.load('./checkpoints/Drawing/sketch_DE_' + key + '.pkl')

        print("load the bin of " + key)
        feature_list = np.fromfile('./checkpoints/bin/man_' + key + '_feature.bin', dtype=np.float32)
        feature_list.shape = 6247, 512
        self.man_list = feature_list
        feature_list = np.fromfile('./checkpoints/bin/female_' + key + '_feature.bin', dtype=np.float32)
        feature_list.shape = 11456, 512
        self.female_list = feature_list
    
    def inference(self, sketch, gender, weight):
        #### refine the hand-drawn sketches
        #gender: 1, man     0, female
        #weight: the weight of project vector, requirement: 0.0 <= weight <= 1.0
        assert(weight >= 0.0 and weight <= 1.0)
        with jt.no_grad():
            sketch_vector = self.drawing_encoder_part(sketch)
            if gender == 1:
                sketch_project = self.get_inter(sketch_vector.numpy(), self.man_list, nearnN=15)
            else:
                sketch_project = self.get_inter(sketch_vector.numpy(), self.female_list, nearnN=15)
            sketch_vector = (1.0 - weight) * sketch_vector + weight * sketch_project

            sketch_geo = self.drawing_decoder_part(sketch_vector)
        return sketch_geo
    
    def get_inter(self, generated_f, feature_list, nearnN=3, w_c=1, random_=-1):
        list_len = jt.array([feature_list.shape[0]])
        b = jt.code([1, nearnN], 
              "int32", [jt.array(feature_list),jt.array(generated_f), list_len], 
        cpu_header="#include <algorithm>",
        cpu_src="""
              using namespace std;
              auto n=out_shape0, k=out_shape1;
              int N=@in2(0);
              
              // use openmp for parallel
              vector<pair<float,int>> id(N);
              #pragma omp parallel for
                for (int j=0; j<N; j++) {
                    auto dis = 0.0;
                    for (int d=0; d<512; d++)
                    {
                      auto dx = @in1(0,d)-@in0(j,d);
                      dis = dis +dx*dx;
                    }
                    id[j] = {dis, j};
                }
                // use c++ lib to sort
                nth_element(id.begin(), 
                  id.begin()+k, id.end());
                // put results in Jittor
                for (int j=0; j<k; j++)
                  @out(0,j) = id[j].second;
              """
        )

        idx_sort = b[0].numpy()

        if nearnN==1:
            vec_mu = feature_list[idx_sort[0]]
            vec_mu = vec_mu * w_c + (1 - w_c) * generated_f
            return vec_mu

        # |  vg - sum( wi*vi )|   et. sum(wi) = 1
        # == | vg - v0 - sum( wi*vi) |   et. w = [1,w1,...,wn]
        A_0 = [feature_list[idx_sort[0],:]]
        A_m = A_0
        for i in range(1,nearnN):
            A_m = np.concatenate((A_m,[feature_list[idx_sort[i],:]]), axis=0)
        
        A_0 = np.array(A_0)
        A_m= np.array(A_m).T
        A_m0 = np.concatenate((A_m[:,1:]-A_0.T, np.ones((1,nearnN-1))*10), axis=0)

        A = np.dot(A_m0.T, A_m0)
        b = np.zeros((1, generated_f.shape[1]+1))
        b[0,0:generated_f.shape[1]] = generated_f-A_0

        B = np.dot(A_m0.T, b.T)

        x = solve(A, B)

        xx = np.zeros((nearnN,1))
        xx[0,0] = 1 - x.sum()
        xx[1:,0] = x[:,0]
        # print(time.time()- start_time)

        vec_mu = np.dot(A_m, xx).T * w_c + (1-w_c)* generated_f
        vec_mu = jt.array(vec_mu.astype('float32'))

        return vec_mu

