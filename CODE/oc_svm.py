import glob         #import all the required modules and library func!
import os
import string
from datetime import datetime
import numpy as np
import itertools
import random
from collections import Counter
import pickle
from sklearn import svm
import pickle as pk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def feature_extraction(filename,rollnum):
    f = open(filename)
    f_list = f.readlines()
    f_list = [x[:-1] for x in f_list]
    if f_list[-1]=='\n':
      f_list.pop()
    
    KeyUps=[]    #Check for KEY UP events 
    for x in f_list:
      if 'KeyUp' in x :
        KeyUps.append(x)
    KeyDowns = [x for x in f_list if 'KeyDown' in x] #Check for keyDOwn Events
   
    time_upkeys =  [item[-26:-2]  for item in KeyUps]   #time Details of the up an down keys event
    time_downkeys =  [item[-26:-2]  for item in KeyDowns]
    
    try:
      pressed_letterup =  [item[-29].upper() for item in KeyUps]       #PRessed Key 
    except: 
      print('Error in ' )
    try:
      pressed_letterdown = [item[-29].upper() for item in KeyDowns]
    except:
      print('Error in ' )
    #print(KeyUps)
    #print(pressed_letterup)
    
    #GEt the latency and Hold time features !
    for i in range(0,len(time_upkeys)-1):
      t = i
      t1 = datetime.strptime(time_downkeys[i], "%d:%m:%Y:%H:%M:%S:%f")
     
      if pressed_letterup[t] != pressed_letterdown[i]:
        j = i
       
        if i == len(time_upkeys)-1:
          j = 0
        while j<len(time_upkeys)-1 and pressed_letterdown[i]!= pressed_letterup[j] and i!=len(time_upkeys)-1:
          j = j+1
        tj = datetime.strptime(time_upkeys[j], "%d:%m:%Y:%H:%M:%S:%f")
        k = i
        
        # Check if keyup is to the left of keydown
        if i == 0:
          k = len(time_upkeys)-1
        while k>=1 and pressed_letterdown[i]!= pressed_letterup[k] and i!=0:
          k = k-1
        tk = datetime.strptime(time_upkeys[k], "%d:%m:%Y:%H:%M:%S:%f")
       
        # Take the minimum of left and right distances if both distances can be valid
        if (tk-t1).total_seconds()>0 and (tj-t1).total_seconds()>0:
          if abs(j-i)<abs(i-k):
            t = j
          else:
            t = k
        elif (tk-t1).total_seconds()<0:
          t = j
        else:
          t = k
      t2 = datetime.strptime(time_upkeys[t], "%d:%m:%Y:%H:%M:%S:%f")
      
      # Latency calculation
      if i!=len(time_upkeys)-1:
        t3 = datetime.strptime(time_downkeys[i+1], "%d:%m:%Y:%H:%M:%S:%f")
        latency = t3-t1
        latency = latency.total_seconds()
        try:
          f = open("Latencies/"+rollnum+pressed_letterdown[i] + pressed_letterdown[i+1] + ".txt", "a+")
        except:
          
          continue
        
        f.write(str(latency) + "\n")	
        f.close()
      
      # Hold time calculation
      hold_time = t2-t1
      hold_time = hold_time.total_seconds()
      try:
        f = open("Hold times/" + rollnum+pressed_letterdown[i] + ".txt", "a+")
      except:
        continue
     
      f.write(str(hold_time) + "\n")	
      f.close()


def common_features():
  f = open("./common_feat.txt", "w+")

  names = list()

  file_count = 9         # (Data of 9 users only available  [5 group 7][4 group 8])
  for filename in glob.glob('Latencies/*'):
    names.append(filename.split('\\')[-1][9:])
  for filename in glob.glob('Hold times/*'):
    names.append(filename.split('\\')[-1][9:])
  
  c = Counter(names)
  single_textnames = [i + '.txt' for i in (string.ascii_uppercase[:26])]
  double_textnames = [i+j+'.txt' for i in (string.ascii_uppercase[:26]) for j in (string.ascii_uppercase[:26])]
  textnames = single_textnames+double_textnames
  
  for name in textnames:
	  if (c[name] == file_count):		# if this number is equal to the number of roll numbers, it is present for all users, and belongs to the common subset
		  f.write("%s\n" % name)		# hence, write out as part of the common subset
  f.close()

def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile, 2):
      if random.randrange(num): continue
      line = aline
    return line

def feature_vector():
  f = open("./common_feat.txt",'r')
  f_list = f.readlines()
  f_list = [x[:-1] for x in f_list]
  if f_list[-1]=='\n':
      f_list.pop()

  rolls = [f.name for f in os.scandir('data') if f.is_dir()]
  random_flist = []
  random_flist2= []
  random_indice = random.sample(range(0, 265), 24)  #24
  random_index2 = random.sample(range(0, 265), 24)        #For Model 2 where diff sets of random features are selected !

  for ind in range(len(random_indice)):
    random_flist.append(f_list[random_indice[ind]])
    random_flist2.append(f_list[random_index2[ind]])

  print(random_flist)
  
  models=1
  for mod in range(models):
    for rollnum in rolls:
      # Extracting the hold times and latencies from the files
      hold_list = glob.glob('Latencies/*')
      lat_list = glob.glob('Hold times/*')
      total_list = hold_list + lat_list
      random_file_list=[]
   
      if mod==0:
        for i in range (len(random_flist)):
          random_file_list.append('Latencies\\'+rollnum+random_flist[i])
          random_file_list.append('Hold times\\'+rollnum+random_flist[i])
      else:
        for i in range (len(random_flist2)):
          random_file_list.append('Latencies\\'+rollnum+random_flist2[i])
          random_file_list.append('Hold times\\'+rollnum+random_flist2[i])
      
      

      total_list = [x for x in random_file_list if x in total_list]
      
      l = []
      feat_vec = np.zeros((256,24))  #24
      # Parsing the files to generate l
      for i in range(256):
          file_no=0
          for x in total_list:            
              inner_list = []
              try:
                  with open(x) as f:
                      line = random_line(f)
                  feat_vec[i,file_no] = line
              except:
                  with open(x) as f:
                      line=random_line(f)
                      feat_vec[i,file_no] = line
              file_no+=1
    
      with open('generated_fvectors/'+rollnum+'_'+str(mod)+'.pickle', 'wb') as f:
          
          pickle.dump(feat_vec, f)

def calc_tp(y_pred_test,Y_test):
  tp=0
  for i in range(len(y_pred_test)):
    if(y_pred_test[i]==Y_test[i] and y_pred_test[i]==1):
      tp=tp+1
  return tp
  

def calc_fp(y_pred_test,Y_test):
  fp=0
  for i in range(len(y_pred_test)):
    if(y_pred_test[i]!=Y_test[i] and y_pred_test[i]==1):
      fp=fp+1
  return fp

def calc_tn(y_pred_test,Y_test):
  tn=0
  for i in range(len(y_pred_test)):
    if(y_pred_test[i]==Y_test[i] and y_pred_test[i]==-1):
      tn=tn+1
  return tn

def calc_fn(y_pred_test,Y_test):
  fn=0
  for i in range(len(y_pred_test)):
    if(y_pred_test[i]!=Y_test[i] and y_pred_test[i]==-1):
      fn=fn+1
  return fn

def calc_metrics(tp,fp,tn,fn):
  precision = (tp*100)/(tp+fp+1e-7)
  recall = (tp*100)/(tp+fn+1e-7)
  specificity = (tn*100)/(tn+fp+1e-7)
  accuracy = 0.5*(tp*100.0/(tp+fn+1e-7)+tn*100.0/(tn+fp+1e-7))
  testing_error = 100-accuracy
  return precision,recall,specificity,accuracy,testing_error

def svm_one_class(X_train,X_test,Y_test):
    nu = np.logspace(-3,0,10)
    
    test_acc = []
    for k in range(len(nu)):
        clf = svm.OneClassSVM(gamma='auto', kernel="rbf", nu = nu[k])
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        n_error_train = y_pred_train[y_pred_train == -1].size     #No of training mislassifiations
        positive_user = y_pred_test[y_pred_test == Y_test].size       #No of Predictions that the user is i
        negative_user = y_pred_test[y_pred_test != Y_test].size       #No of Predictions that the user is not i
        tp=calc_tp(y_pred_test,Y_test)
        fp=calc_fp(y_pred_test,Y_test)
        tn=calc_tn(y_pred_test,Y_test)
        fn=calc_fn(y_pred_test,Y_test)
        precision,recall,specificity,accuracy,testing_error = calc_metrics(tp,fp,tn,fn)
        print ("Nu : %f" %nu[k])
        print("Training Dataset Size : %s" %y_pred_train.size)
        print("No of Training Misclassifications : %s" %n_error_train)
        print("Training Error : %.2f percent" %((n_error_train*100)/y_pred_train.size))
        print("Testing Dataset Size : %s" %Y_test.size)
        print("Correct Decissions: %s" %positive_user)
        print("Incorrect Decissions: %s" %negative_user)
        print("Testing Accuracy: %.2f percent" %(accuracy))
        print("Testing Error: %.2f percent" %(testing_error))
        print("Precision: %.2f percent" %(precision))
        print("Recall: %.2f percent" %(recall))
        print("Specificity: %.2f percent" %(specificity))
        print()
        test_acc.append(accuracy)
    return accuracy,precision, recall, test_acc

def plot(average_test_acc,roll):
    nu = np.logspace(-3,0,10)
    
    ax = plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(11))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(100))
    ax.yaxis.set_major_locator(plt.MaxNLocator(11))
    ax.yaxis.set_minor_locator(plt.MaxNLocator(100))

    plt.xscale("log")       
    plt.plot(nu,average_test_acc, label = roll)
    ax.grid(which = 'major', linestyle='-', linewidth = 0.9, alpha=1.0)
    ax.grid(which = 'minor', linestyle=':', linewidth = 0.6, alpha=0.8)
    plt.xticks(rotation='vertical')
    plt.xlabel("Nu")
    plt.ylabel("average accuracy")
    plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.25), ncol=4)
    plt.savefig( "plots/" + roll + '.png',dpi=500, bbox_inches = 'tight')

def main():
    for name in glob.glob('data/*/*.txt'):
        roll_num=name.split('\\')[-2]
        feature_extraction(name,roll_num)
    common_features()
    feature_vector()
    n = 0     
    data=[]    #Data Set
    roll=[]

    rollnums=[f.name for f in os.scandir('data') if f.is_dir()]
    for rollnum in rollnums:
        n=n+1
        roll.append(rollnum)
        with open('generated_fvectors/'+rollnum+'_0.pickle', "rb") as input_file:
            ef = pk.load(input_file)
            ck=[]
            for item in ef:
                ck.append(item)
            data.append(np.array(ck))


    for i in range(n):	# n is the number of roll numbers (users)
        #concatinate all the list together for testing as all column size same
        if(i==0):
            test=data[1]
            for k in range (2,n):
                test=np.vstack((test,data[k]))                     #Stack the datasets together
        else:
            test=data[0]
            for k in range (1,n):
                if(k!=i):
                    test=np.vstack((test,data[k]))                 #Stack the datasets together

        print("-------------------------------------------------------")
        print("For User %s" %roll[i])
        total_test_acc = []
        average_test_acc = []
        for j in range (5):
                                                    #5 Fold cross validation
            print("Validation Fold %s:" %(j+1))
            data_train=data[i]
                  
            np.random.shuffle(data_train)
            sz=int(0.2*data_train.shape[0])+1
            
            data_train_final=data_train[:-sz]                     #Discarding last 5% entries for training and add them in tesing
            data_test_final=data_train[-sz:]
    	    
            scaler = StandardScaler()
            data_train_final = scaler.fit_transform(data_train_final)

            pca = PCA(n_components = 8)
            data_train_final = pca.fit_transform(data_train_final)
            y=np.ones(sz)


            sz=int(0.05*data_train.shape[0])+1   
            #print(sz)           #No of test sample needed more for 80:20 ratio 80 authentic (same user) and 20 other users
            np.random.shuffle(test)                               
            data_test_final=np.vstack((data_test_final,test[:sz]))
            data_test_final = scaler.transform(data_test_final)
            data_test_final = pca.transform(data_test_final)
            z=-np.ones(sz)
            y_final=np.append(y,z)                       #Result vector for Testing
            accuracy,precision, recall, test_acc = svm_one_class(data_train_final,data_test_final,y_final)       # calling SVM function
            total_test_acc.append(test_acc)

          
        for k in range(len(test_acc)):
            average = 0
            for j in range(5):
                average = total_test_acc[j][k]+ average
            average = average/5
            average_test_acc.append(average)
        plot(average_test_acc,roll[i])

if __name__=="__main__":
  main()