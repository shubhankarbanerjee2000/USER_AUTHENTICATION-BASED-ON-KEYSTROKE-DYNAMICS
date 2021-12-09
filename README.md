# USER_AUTHENTICATION-BASED-ON-KEYSTROKE-DYNAMICS
Machine Intelligence and Expert Systems IIT KGP term project . The aim is to perform user authentication using keystroke dynamics via One Class SVM classification
USER AUTHENTICATION USING KEYSTROKE DYNAMICS :


STEPS TO RUN :
METHOD 1 : USING GOOGLE COLAB 
* Run the .ipynb file as provided in the submission with google colab
Link to notebook : https://colab.research.google.com/drive/1qJ0Bp8kKknlHn9DEJsHEJkrbFnrDQxDe?usp=sharing
1.Use the given link to the data folder for executing the code. Click on the link and add the ‘data’ folder as a shortcut to your drive. You can also download the folder and upload it in your google drive.
https://drive.google.com/drive/folders/1TnHlcLsHW10mW-eHaMIyjspgJ8_mSMh-?usp=sharing
 Just make this data accessible to your google drive .
2. Run the first code cell to mount your google drive 
3. From the files section of google colab create new folders and files named -
Folders-
        Hold time
        Latencies
        generated_fvectors
        plots
Files-
       common_feat.txt
4. Run the cells normally to get the results


These Cells have the main code all combined together . Thus running the cell would be sufficient to get the results .
The code in the one_class_svm function can be uncommented to change to different parameter variations .  Also the Type of kernel used can be varied when the oneclassSVM is called under this function . The type could be changed to “linear” , “polynomial” or “rbf”.




METHOD 2: RUN THE .PY FILE LOCALLY


1. Unzip the mies_code folder, it also contains the data, so no need to download it.
2. Open command prompt and navigate to the mies_code directory (using cd command)
3. Run this command: python oc_svm.py
4. The classification results will be printed on screen . Corresponding to each roll number we do 5 fold cross validation and for each validation, we obtain predictions for 10 values of nu. 
5. The plots shown in the report are obtained under the plots folder.
