#include <iostream>

#include "arczee.h"
//#include "cldev.h"

using namespace std;


#define STRINGIFY(src) #src

//#include "dpotrf.cl"
extern char * chol_str;


void arczee_init(cldev &cd)
{
    vector<std::pair<string,string> > kernelContents;
    vector<string> kernelFiles;
    vector<string> kernelNames;
    const char *str;



    //*****initialize kernel from string
    //kernelContents.push_back(std::make_pair<string, string>("chol", chol_str));
    //kernelContents.push_back(std::make_pair<string, string>("autodiff", autodiff_Str()));

    //cd.createKernelsFromStr(kernelContents, kernelNames);

    //*****initialize kernel from files
	kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\include\\cldef.h");
	kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\common.cl");
	/*
	kernelNames.push_back("chol_v1_step1");
	kernelNames.push_back("chol_v1_step2");
	kernelNames.push_back("chol_v1_step3");
    kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrf_v1.cl");
	*/

	//kernelNames.push_back("chol_v2_step1");
	//kernelNames.push_back("chol_v2_step2");
	//kernelNames.push_back("chol_v2_step3");
	//kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrf_v2.cl");

	kernelNames.push_back("chol_v3_step1");
	kernelNames.push_back("chol_v3_step2");
	kernelNames.push_back("chol_v3_step3");
	kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrf_v3.cl");


	kernelNames.push_back("chol_v4_step1");
	kernelNames.push_back("chol_v4_step2");
	kernelNames.push_back("chol_v4_step3");
	kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrf_v4.cl");

	kernelNames.push_back("dpotrs_v1_p1_s1");
	kernelNames.push_back("dpotrs_v1_p1_s2");
	kernelNames.push_back("dpotrs_v1_p1_s3");
	kernelNames.push_back("dpotrs_v1_p2_s1");
	kernelNames.push_back("dpotrs_v1_p2_s2");
	kernelNames.push_back("dpotrs_v1_p2_s3");
    kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrs_v1.cl");

	kernelNames.push_back("dpotrs_v2_p1");
	kernelNames.push_back("dpotrs_v2_p2");
	kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrs_v2.cl");

	kernelNames.push_back("dpotrs_v3_p1_s1");
	kernelNames.push_back("dpotrs_v3_p1_s2");
	kernelNames.push_back("dpotrs_v3_p2_s1");
	kernelNames.push_back("dpotrs_v3_p2_s2");
	kernelFiles.push_back("E:\\sync_directory\\workspace\\ArcZeeMath\\src\\ArcZeeMath\\dpotrs_v3.cl");

    cd.createKernels(kernelFiles, kernelNames);




}
