## Description
"Tool for on-line dynamic security assessment" developed in WP4 in task T4.6.
It implements a machine learning approach based on an Artificial Neural Networks (ANNs) to perform on-line dynamic security assessment with respect to frequency stability in future power systems characterized by large shares of converter interfaced generation, namely Renewable Energy Sources (RES) such as wind and solar. 
It is suitable to be run either on-line or off-line (e.g., for day-ahead operational planning purposes.

## How to run
The tool can be executed from a command line prompt by running the corresponding Python script through the command “Running_DSA_tool_main.py”.

It requires installation of:
*	Python version 3.7 or higher(Python 3.9 was used to developed and test the tool) for Windows OS with the following additional Python packages/libraires installed:
	*	NumPy (1.19.5 release)
	*	TensorFlow (2.5.0 release)
	*	Keras API (provided within the TensorFlow library)


## Inputs
The tool receives and processes internally several inputs read from the following three files (which must be located in the same directory of the script “Running_DSA_tool_main.py”): 
* “ANN_id.h5” – this file is an encrypted file (TensorFlow/Keras object) that contains the model of an ANN already trained for given contingency.  
* “config.csv” – contains information related with control parameters of the tool and data required for the standardization method, which is run during the tool execution. In particular, the following data is included in this file and must be set: number of the Operating Points (OPs) to be run/ tested, number of ANN to be run, number of ANN inputs variables, number of ANN output variables, and maximum and minimum values of each ANN input variable (gathered from the training dataset) to be used by the standardization method.
* “test_cases.csv” – this file contains the values of the variables selected to be used as input of the ANN model correspondent to the OPs to be tested. For OP to be run the input varaibles are:
	* Input Var1,SGs_Total_Pgen - Total synchronous machines active power production (MW)
	* Input Var2,WFs_Total_Pgen - Total RES (Wind) active power production (MW)
	* Input Var3,H_SC1_Bus34 - Synchronous Condenser (SC) inertia for the SC with id 1 and located in bus 34 (s)
	* Input Var4,H_SC2_Bus37 - Synchronous Condenser (SC) inertia for the SC with id 2 and located in bus 37 (s)
	* Input Var5,H_SC3_Bus38 - Synchronous Condenser (SC) inertia for the SC with id 3 and located in bus 38 (s)
	* Input Var6,Total_H_SMs - Total synchronous machines interia without considering SCs(s)


## Outputs
After running the tool, it is displayed on the screen information about the total running time (“Elapsed time”) and a message informing the user to check the “results.csv” file where all the results are compiled and can be properly analyzed. 
The results include for each Operating Point (OP) the outputs variables of the tool (i.e., the estimated values for the frequency establity indicators nadir and RoCOF), information regarding system security (system is or not secure), and results about the SCs that needed to be turned on in case of the system is unsecure.

## Error Codes:
The following code errors (non-related with Python exceptions) are returned by the tool: 
*	Code 0 – “Process finished with no errors. The tool run successfully for N ANN (N Contingencies) and X OPs”, where N is replaced by the number of ANN/contingencies that were run in parallel by the tool and X replaced by the number of OPs tested";
*	Code 1 – “Error: “ANN_id.h5” file is missing. Please verify its name and ensure that its path corresponds to the directory where tool Python script is located";
*	Code 2 – “Error: “Config.csv” file is missing. Please verify its name and ensure that its path corresponds to the directory where tool Python script is located”;
*	Code 3 – “Error: “Test_cases.csv” file is missing. Please verify its name and ensure that its path corresponds to the directory where tool Python script is located”;
*	Code 4 – “Error: Bad, insufficient or inconsistent data detected in “Config.csv” file. Please revise the whole file carefully including data format and values”;
*	Code 5 – “Error: Bad, insufficient or inconsistent data detected in “Test_cases.csv” file. Please revise the whole file carefully including data format and values”;
*	Code 6 – “An unexpected error occurred. Please contact the tool developer”.

## Addtional information:
In addition to the input files required for executing the tool, a complementary informative Excel file (complementary_information.xlsx”) is also provided. 
This optional file contains information related with the characteristics of test system, the critical contingencies/disturbances simulated, the ANN model/architecture, and related with the training process of the tool, among other information that can be of interest and very usefull to the user / TSOs.