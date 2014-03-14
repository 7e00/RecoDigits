#include <iostream>
#include <fstream>
#include <ctime>

#include "ferns.h"

using namespace std;
ofstream logfile("log", ios::out);

int getint(char *str, int len = 4)
{
    unsigned char *ustr = (unsigned char *)str;
    int res = 0;
    for(int i = 0; i < len; i++)
        res = res*256 + ustr[i];
    return res;
}

const char *trainfile = "train-images.idx3-ubyte";
const char *labelfile = "train-labels.idx1-ubyte";
const char *testfile = "t10k-images.idx3-ubyte";
const char *testlabel = "t10k-labels.idx1-ubyte";
struct TData
{
    double *data;
    int *ys;
    int N;
    int F;
};
TData getData(const char *file, const char *labelfile, int maxnum = 100000)
{
    TData res = {0, 0, 0, 0};
    ifstream tf(file, ios::in | ios::binary);
    ifstream lf(labelfile, ios::in | ios::binary);
    char magic[4] = {0};
    tf.read(magic, 4);
    if(magic[2] != 8 || magic[3] != 3)
    {
        logfile << "not a valid data file!" << endl;
        return res;
    }
    lf.read(magic, 4);
    if(magic[2] != 8 || magic[3] != 1)
    {
        logfile << "not a valid label file!" << endl;
        return res;
    }
    int tdnum, nrows, ncols, labelnum;
    tf.read(magic, 4);
    tdnum = getint(magic);
    lf.read(magic, 4);
    labelnum = getint(magic);
    if(tdnum != labelnum)
    {
        logfile << "not right label file for this data set!" << endl;
        return res;
    }
    tf.read(magic, 4);
    nrows = getint(magic);
    tf.read(magic, 4);
    ncols = getint(magic);
    int tnum = min(maxnum, tdnum);
    double *data = new double[tnum*nrows*ncols];
    int *ys = new int[tnum];
    int F = nrows * ncols;
    unsigned char *imgarr = new unsigned char[nrows*ncols];
    unsigned char label;
    logfile << "read data, total " <<tnum<<" entries..." << endl;
    for(int i = 0; i < tnum; i++)
    {
        double *entry = data + i*F;
        tf.read((char *)imgarr, F);
        for (int j = 0; j < F; ++j)
            entry[j] = (double)imgarr[j] / 255.0;
        lf.read((char *)(&label), 1);
        ys[i] = label;
        //logfile << "entry " << i << " label " << ys[i] << endl;
    }
    res.data = data;
    res.ys = ys;
    res.N = tnum;
    res.F = F;
    delete []imgarr;
    return res;
}

int main()
{
    srand(time(0));
    logfile<<"load train data..."<<endl;
    TData traindata = getData(trainfile, labelfile);
    if(traindata.N == 0)
        return 1;
    RandomFerns rf(200, 12);
    Diff_Binary_feature dbf(2400, traindata.F, 0, 1);
    logfile << "read train data over, "<<traindata.N<<" entries" << endl;
    logfile <<"begin train..."<<endl;

    logfile << "train over, correct rate is "<<rf.train(traindata.data, traindata.ys, traindata.N, traindata.F, 10, &dbf)<<endl;
    delete []traindata.data;
    delete []traindata.ys;

    logfile<<"load test data..."<<endl;
    TData testdata = getData(testfile, testlabel);
    logfile << "read test data over, "<<testdata.N<<" entries" << endl;
    int *ysout = new int[testdata.N];
    logfile <<"begin test..."<<endl;
    logfile <<"test over, correct rate is "<<rf.evaluate(testdata.data, testdata.ys, testdata.N, testdata.F, ysout)<<endl;
    logfile << "compare ...\nlabel\tout" << endl;
    int err = 0;
    for(int i = 0; i < testdata.N; i++)
    {
        logfile<<testdata.ys[i]<<'\t'<<ysout[i];
        if(testdata.ys[i] != ysout[i])
        {
            logfile<<"\te";
            err++;
        }
        logfile<<endl;
    }
    logfile<<"err is "<<err<<endl;
    delete []testdata.data;
    delete []testdata.ys;
    delete []ysout;
    return 0;
}
