#include<iostream>
#include<vector>
#include<time.h>
#include<omp.h>
#include <xmmintrin.h>
#include <nmmintrin.h> // for SSE4.2
#include <immintrin.h> // for AVX
using namespace std;
/*
实现一个包含max_pooling操作和张量求和操作的计算图的前馈过程
Formulation : dst[32,64,56,56] = add(max_pooling(src1[32,64,112,112]), src2[32,1,56,56])
*/

//定义Tensor为四维矩阵[b, c, h, w]
typedef vector<vector<vector<vector<int> > > > Tensor;

//定义一个函数获得四维tensor[b, c, h, w]
Tensor getTensor(int batch, int channels, int height, int width, int init){
    vector<int> w(width, init);
    vector<vector<int> > h(height, w);
    vector<vector<vector<int> > > c(channels, h);
    Tensor t(batch, c);
    return t;
}

//max_pooling函数
Tensor maxPooling(Tensor& t, vector<int> kernel_size, vector<int> pad, vector<int> stride){
    int batch_size = t.size();
    int channels = t[0].size();
    int height = t[0][0].size();
    int width = t[0][0][0].size();

    int out_height = (height + 2*pad[0] - kernel_size[0]) / stride[0] + 1;
    int out_width = (width + 2*pad[1] - kernel_size[1]) / stride[1] + 1;
    Tensor output = getTensor(batch_size, channels, out_height, out_width, 0);
    
    //先进行填充,原始 t resize成[b, c, height+2*pad, width+2*pad] 
    //使用OpenMP
    #pragma omp parallel for
    for(int b=0; b<batch_size; b++){
        for(int c=0; c<channels; c++){
            t[b][c].resize(height + 2*pad[0]);
            for(int h=0; h<height+2*pad[0]; h++){
            	t[b][c][h].resize(width + 2*pad[1]);
			}
        }
    }
	 
    //max_pooling功能， 通过循环体求h,w维度上一个 kernel_size邻域内的最大值
    #pragma omp parallel for
    for(int b=0; b<batch_size; b++){
        for(int c=0; c<channels; c++){
            for(int h=0; h<out_height; h++){
                for(int w=0; w<out_width; w++){
                    //滑动窗口的起始位置
                    int sx = h*stride[0];
                    int sy = w*stride[1];
                    int max_element = t[b][c][sx][sy];

                    //内层二重循环寻找最大
                    for(int i=0; i<kernel_size[0]; i++){
                        for(int j=0; j<kernel_size[1]; j++){
                            max_element = max(t[b][c][sx+i][sy+j], max_element);
                        }
                    }
                    output[b][c][h][w] = max_element;
                }
            }
        }
    }
    return output;
}

//tensor_add函数
//类似numpy broadcast， 从后往前匹配，维度一致或者其中一个为1才能broadcast, 这里没有实现维数不匹配的情况
Tensor tensorAdd(Tensor& t1, Tensor& t2){
    //[b, c, h, w]维度
    int b1 = t1.size(), c1 = t1[0].size(), h1 = t1[0][0].size(), w1 = t1[0][0][0].size();
    int b2 = t2.size(), c2 = t2[0].size(), h2 = t2[0][0].size(), w2 = t2[0][0][0].size();
    int res_b = max(b1, b2), res_c = max(c1, c2), res_h = max(h1, h2), res_w = max(w1, w2);
    Tensor resTensor = getTensor(res_b, res_c, res_h, res_w, 0);
    //维度检查
    if(w1 != w2 && w1 != 1 && w2 != 1){
        cout<<"Dimension not matched"<<endl;
        return resTensor;
    }
    if(h1 != h2 && h1 != 1 && h2 != 1){
        cout<<"Dimension not matched"<<endl;
        return resTensor;
    }
    if(c1 != c2 && c1 != 1 && c2 != 1){
        cout<<"Dimension not matched"<<endl;
        return resTensor;
    }
    if(b1 != b2 && b1 != 1 && b2 != 1){
        cout<<"Dimension not matched"<<endl;
        return resTensor;
    }
    //矩阵相加
    //使用SIMD 单指令多数据处理来优化矩阵加操作，将多个操作打包到寄存器中，使用四个线程
    #pragma omp parallel for 
    for(int b=0; b<res_b; b++){
        int b_1 = min(b, b1-1);
        int b_2 = min(b, b2-1);
        for(int c=0; c<res_c; c++){
            int c_1 = min(c, c1-1);
            int c_2 = min(c, c2-1);
            for(int h=0; h<res_h; h++){
                int h_1 = min(h, h1-1);
                int h_2 = min(h, h2-1);
                //当最后一个维度均为1
                if(w1 == 1 && w2 == 1){
                    resTensor[b][c][h][0] = t1[b_1][c_1][h_1][0] + t2[b_2][c_2][h_2][0];
                }else{
                    //分组，取模操作将剩余的w划分为一组
                    int init_w = res_w / 4;
                    int extra_w = res_w % 4;
                    __m128i add;
                    for(int w=0; w<init_w; w++){
                        if(w1 == 1){
                            //打包tensor1
                            __m128i x1 = _mm_setr_epi32(t1[b_1][c_1][h_1][0],
                            t1[b_1][c_1][h_1][0],
                            t1[b_1][c_1][h_1][0],
                            t1[b_1][c_1][h_1][0]);
                            //打包tensor2
                            __m128i x2 = _mm_setr_epi32(t2[b_2][c_2][h_2][0+4*w],
                            t2[b_2][c_2][h_2][1+4*w],
                            t2[b_2][c_2][h_2][2+4*w],
                            t2[b_2][c_2][h_2][3+4*w]);
                            add = _mm_add_epi32(x1, x2);
                            _mm_storeu_si128((__m128i*)(&resTensor[b][c][h][4*w]), add);
                        }else if(w2 == 1){
                            //打包tensor1
                            __m128i x1 = _mm_setr_epi32(t1[b_1][c_1][h_1][0+4*w],
                            t1[b_1][c_1][h_1][1+4*w],
                            t1[b_1][c_1][h_1][2+4*w],
                            t1[b_1][c_1][h_1][3+4*w]);
                            //打包tensor2
                            __m128i x2 = _mm_setr_epi32(t2[b_2][c_2][h_2][0],
                            t2[b_2][c_2][h_2][0],
                            t2[b_2][c_2][h_2][0],
                            t2[b_2][c_2][h_2][0]);
                            add = _mm_add_epi32(x1, x2);
                            _mm_storeu_si128((__m128i*)(&resTensor[b][c][h][4*w]), add);
                        }else{
                            //打包tensor1
                            //cout<<"case 1"<<endl;
                            __m128i x1 = _mm_setr_epi32(t1[b_1][c_1][h_1][0+4*w],
                            t1[b_1][c_1][h_1][1+4*w],
                            t1[b_1][c_1][h_1][2+4*w],
                            t1[b_1][c_1][h_1][3+4*w]);
                            //cout<<"case 2"<<endl;
                            //打包tensor2
                            __m128i x2 = _mm_setr_epi32(t2[b_2][c_2][h_2][0+4*w],
                            t2[b_2][c_2][h_2][1+4*w],
                            t2[b_2][c_2][h_2][2+4*w],
                            t2[b_2][c_2][h_2][3+4*w]);
                            add = _mm_add_epi32(x1, x2);
                            //cout<<"case 3"<<endl;
                            _mm_storeu_si128((__m128i*)(&resTensor[b][c][h][4*w]), add);
                            //cout<<"case 4"<<endl;
                        }
                    }
                    //若有剩余的group, 单独计算加法
                    if(extra_w){
                        for(int ew=0; ew<extra_w; ew++){
                            int ew1 = init_w*4 + ew;
                            int ew2 = init_w*4 + ew;
                            if(w1 == 1){
                                resTensor[b][c][h][init_w*4 + ew] = t1[b_1][c_1][h_1][0] + t2[b_2][c_2][h_2][ew2];
                            }else if(w2 == 1){
                                resTensor[b][c][h][init_w*4 + ew] = t1[b_1][c_1][h_1][ew1] + t2[b_2][c_2][h_2][0];
                            }else{
                                resTensor[b][c][h][init_w*4 + ew] = t1[b_1][c_1][h_1][ew1] + t2[b_2][c_2][h_2][ew2];
                            }
                        }
                    }
                }
            }
        }
    }
    return resTensor;
}

int main(){
    Tensor src1 = getTensor(32, 64, 112, 112, 0);
    Tensor src2 = getTensor(32, 1, 56, 56, 0);
    vector<int> ks = {3, 3};
    vector<int> pad = {1, 1};
    vector<int> s = {2, 2};
    auto start = clock();
    Tensor mpRes = maxPooling(src1, ks, pad, s);
    Tensor addRes = tensorAdd(mpRes, src2);
    auto end = clock();
    cout<<"Result Shape: "<<addRes.size()<<" "<<addRes[0].size()<<" "<<addRes[0][0].size()<<" "<<addRes[0][0][0].size()<<endl;
    cout<<"Time Consume: "<<(double)(end-start)<<" ms "<<endl;
    return 0;
}
