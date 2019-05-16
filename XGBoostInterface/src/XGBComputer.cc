#include "XGBoostCMSSW/XGBoostInterface/interface/XGBComputer.h"

XGBComputer::XGBComputer(mva_variables* vars, std::string model_file)
{
    vars_ = vars;
    //---load the model
    XGBoosterCreate(NULL, 0, &booster_);
    XGBoosterLoadModel(booster_, model_file.c_str());
};

std::vector<float> XGBComputer::operator() ()
{    
    float values[1][vars_->size()];

    int ivar=0;
    for(auto& var : *vars_)
    {
        values[0][ivar] = std::get<1>(var);
        ++ivar;
    }

    DMatrixHandle dvalues;
    XGDMatrixCreateFromMat(reinterpret_cast<float*>(values), 1, vars_->size(), 0., &dvalues);
    
    bst_ulong out_len=0;
    const float* score;

    auto ret = XGBoosterPredict(booster_, dvalues, 0, 0, &out_len, &score);

    XGDMatrixFree(dvalues);

    std::vector<float> results;
    if(ret==0)
    {
        for(unsigned int ic=0; ic<out_len; ++ic)
            results.push_back(score[ic]);
    }
    
    return results;    
}
