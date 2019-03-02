#include <stdio.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
void CVMat_BGR_to_vector(cv::Mat& img, std::vector<float> &output)
{
    for(int i=0; i<img.rows; ++i)
    {
        //获取第i行首像素指针
        cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
//        Vec3b *p2 = image.ptr<Vec3b>(i);
        for(int j=0; j<img.cols; ++j)
        {
            //将img的bgr转为image的rgb  并转化维fp32
            output[i * img.cols * img.channels() + j * img.channels() + 2] = p1[j][0] / 255.0;
            output[i * img.cols * img.channels() + j * img.channels() + 1] = p1[j][1] /255.0;
            output[i * img.cols * img.channels() + j * img.channels() + 0] = p1[j][2] /255.0;
        }
    }
}

std::vector<std::string>  name =
{
    "daisy",
    "tulips",
    "dandelion",
    "sunflowers",
    "roses"
};
class TFLiteRunner
{
public:
    TFLiteRunner(std::string& modelFile, std::string &inputname,std::string &outputName):
        m_modelFile(modelFile),m_inputname(inputname),m_outputName(outputName)
    {
        Init();
    }
    virtual ~TFLiteRunner() {}

public:
    virtual bool Init()
    {
        // Load model
        //std::unique_ptr<tflite::FlatBufferModel> model;
        model = tflite::FlatBufferModel::BuildFromFile(m_modelFile.c_str());
        if (model == nullptr)
        {
            std::cerr << "failed to load tflite model !" << std::endl;
            return false;
        }

        // Build the interpreter
        //tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (interpreter == nullptr)
        {
            std::cerr << "failed to build tflite interpreter !" << std::endl;
            return false;
        }
        bool verbose = true;
        if (verbose)
        {
            std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
            std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
            std::cout << "number of inputs: " << interpreter->inputs().size() << "\n";
            std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
            std::cout << "number of outputs: " << interpreter->outputs().size() << "\n";
            std::cout << "output(0) name: " << interpreter->GetOutputName(0) << "\n";

            int t_size = interpreter->tensors_size();
            for (int i = 0; i < t_size; i++)
            {
                if (false && interpreter->tensor(i)->name)
                    std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                              << interpreter->tensor(i)->bytes << ", "
                              << interpreter->tensor(i)->type << ", "
                              << interpreter->tensor(i)->params.scale << ", "
                              << interpreter->tensor(i)->params.zero_point << "\n";
            }
        }
        interpreter->SetNumThreads(4);

        // Allocate tensor buffers.
        if (interpreter->AllocateTensors() != kTfLiteOk)
        {
            std::cerr << "failed to allocate tflite tensor buffers !" << std::endl;
            return false;
        }
        std::cout << "=== Pre-invoke Interpreter State ===" << std::endl;
        //tflite::PrintInterpreterState(interpreter.get());

        // get input dimension from the input tensor metadata
        // assuming one input only
        int input = interpreter->inputs()[0];
        TfLiteIntArray* dims = interpreter->tensor(input)->dims;
        int wanted_height, wanted_width, wanted_channels;

        if (dims->size == 4)
        {
            wanted_height = dims->data[1];
            wanted_width = dims->data[2];
            wanted_channels = dims->data[3];
        }
        else if(dims->size == 3)
        {
            wanted_height = dims->data[0];
            wanted_width = dims->data[1];
            wanted_channels = dims->data[2];
        }
        else
        {
            std::cerr << "input tensor's dims->size=" << dims->size << " is error !" << std::endl;
            return false;
        }
        std::cout << "wanted_height = " << wanted_height << std::endl;
        std::cout << "wanted_width = " << wanted_width << std::endl;
        std::cout << "wanted_channels = " << wanted_channels << std::endl;


        // Read output buffers
        int output = interpreter->outputs()[0];
        TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];

        std::cout << "output = " << output << "   "<<interpreter->outputs().size()<<std::endl;
        std::cout << "output_size = " << output_size << std::endl;

        return true;
    }



    virtual bool inference(cv::Mat &simg, int label)
    {
        if(false)
        {

            std::cout <<std::endl;
            std::cout <<std::endl;
            std::cout <<std::endl;

            int input_tensor_ix  = interpreter ->inputs()[0];
            auto type = interpreter ->tensor(input_tensor_ix)->type;
            std::cout << "input tensor index: " << input_tensor_ix << "\n";
            std::cout << "input tensor type: " << type << "\n";

            //auto input = interpreter ->typed_tensor<float>(input_tensor_ix);
            std::cout << "input(0) name: " << interpreter ->GetInputName(0) << "\n";

            TfLiteIntArray* dims = interpreter ->tensor(input_tensor_ix)->dims;
            int batch_size = dims->data[0];
            int wanted_height = dims->data[1];
            int wanted_width = dims->data[2];
            int wanted_channels = dims->data[3];

            std::cout << batch_size << " " <<  wanted_height << " " << wanted_width << " " << wanted_channels << "\n";
            int output_tensor_ix  = interpreter ->outputs()[0];
            auto out_type = interpreter ->tensor(output_tensor_ix)->type;
            std::cout << "output tensor type: " << out_type << "\n";

            std::cout <<std::endl;
            std::cout <<std::endl;
            std::cout <<std::endl;
        }

        //std::cout << "outpu+++++++111++++++++++++: "  << "\n";
        cv::Mat img;
        cv::resize(simg,img,cv::Size(128,128));
        int nWidth = img.cols;
        int nHeight=img.rows;
        int nChannel = img.channels();
        int num_pixels = nWidth*nHeight*nChannel;

        std::vector<float> pInputBuff(num_pixels);
        CVMat_BGR_to_vector(img,pInputBuff);

        float* input = interpreter ->typed_input_tensor<float>(0);

        memcpy(input, pInputBuff.data(), sizeof(float)*num_pixels);

        //std::cout << "outpu+++++++++++++++++++: "  << "\n";

        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cout << "failed to inference tflite !" << std::endl;
            return false;
        }
        // std::cout << "outpu-------------------- "  << "\n";


        int nOutSize = interpreter->outputs().size();
        for(int i = 0; i < nOutSize; i++)
        {
            auto probs = interpreter ->typed_output_tensor<float>(i);
            int output = interpreter->outputs()[i];

            TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
            // assume output dims to be something like (1, 1, ... ,size)
            auto output_size = output_dims->data[output_dims->size - 1];

            std::vector<float>result1{probs, probs + output_size};
            std::cout << interpreter->GetOutputName(i) << " " << output_size << std::endl;

            auto maxPosition = std::max_element(result1.begin(), result1.end());

            int class_id = maxPosition - result1.begin();
            cv::Scalar color = cv::Scalar(0, 255, 0);
            if (label == class_id)
            {
                color = cv::Scalar(0, 255, 0);
            }
            else
            {
                color = cv::Scalar(0, 0, 255);
            }
            cv::putText(simg, "Pre: "+(name[label]) + " true: "+(name[class_id]), cv::Point(10,50), cv::FONT_HERSHEY_COMPLEX, 0.6, color, 1, CV_AA);
            cv::putText(simg, "Prob: "+std::to_string(result1[class_id]), cv::Point(10,70), cv::FONT_HERSHEY_COMPLEX, 0.6, color, 1, CV_AA);

            //result[interpreter ->GetOutputName(i)] = result1;
        }

        return true;
    }

    virtual bool UnInit()
    {
        return true;
    }
protected:

private:
    std::string& m_modelFile;
    std::string &m_inputname;
    std::string &m_outputName;

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::FlatBufferModel> model;

};

void split(const std::string &sourceSrt, std::string& filepath,int &nType )
{
    std::string::size_type found = sourceSrt.find(' ');
    filepath = sourceSrt.substr(0,found);
    nType = atoi(sourceSrt.substr(found).c_str());
}

struct Info
{
    int nType;
    std::string filepath;
};
int main(int argc, char *argv[])
{
    /*
    std::string modelFile = "/home/jerry/Desktop/Code/classify_factory/model/classify_cnn/classify_cnn.tflite";
    std::string inputname = "input_2";
    std::string outputName= "dense_3/Softmax";
    */
    std::string modelFile = "/home/jerry/Desktop/Code/classify_factory/model/classify_MobileNet/classify_MobileNet.tflite";
    std::string inputname = "input_2";
    std::string outputName= "reshape_2/Reshape";

    std::string testfile = "/home/jerry/data/data/classify/flower_photos/val_list.txt";
    std::ifstream infile;
    infile.open(testfile.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行

    std::string source;
    std::vector<Info> tInfoList;
    while(getline(infile,source))
    {
        Info tInfo;
        split(source,tInfo.filepath,tInfo.nType);
        tInfoList.emplace_back(tInfo);
    }

    TFLiteRunner tTFLiteRunner(modelFile,inputname,outputName);
    tTFLiteRunner.Init();

    int key = 0;
    auto ite = tInfoList.begin();
    while(true)
    {
        const Info& tInfo = *ite ;

        cv::Mat imag = cv::imread(tInfo.filepath);
        tTFLiteRunner.inference(imag,tInfo.nType);
        cv::imshow("test",imag);
        key = cv::waitKey(0);
        if(key == 'q')
            break;
        ite++;
        if(ite == tInfoList.end())
        {
            ite = tInfoList.begin();
        }
    }

    return 0;
}
