#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <iostream>

#include <xgboost/c_api.h>

class XGBComputer
{       
public:
    typedef std::vector<std::tuple<std::string, float> > mva_variables;
    
    //--ctros---
    XGBComputer() {};
    XGBComputer(mva_variables* vars, std::string model_file);

    //---dtor---
    ~XGBComputer()
        {
        };
    
    //---setters---
    void SetVariables(mva_variables* vars) { vars_ = vars; };

    //---getters---
    std::vector<float> operator() ();

private:
    mva_variables* vars_;
    BoosterHandle  booster_;
};
