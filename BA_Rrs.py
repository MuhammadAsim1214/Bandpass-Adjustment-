# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:23:56 2023

@author: muhammad.asim@uit.no/muhammad.asim.1214@gmail.com
"""

# Import required libraries
import numpy as np 
import matplotlib.lines as mlines
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy import stats

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
tf.reset_default_graph()
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import seaborn as sns





Pre_Data =[]
OLI_Data=[] 
MSI_Data=[]
Linear_P=[]
R2_Test_All=[]

# Excel File where OLI and MSi Rrs data are provided in columns

excel_file = 'C:/Users/./Bandpass_Adjustment/Excel_File_BA.xlsx'
file1 = pd.read_excel(excel_file,sheet_name='Sheet1')  

# Bandpass Adjustment of MSI 443nm band. 
OLI_Band = 'Rrs_443_L8_OC'
MSI_Band=  'Rrs_443_S2_OC'

#x_vals_train1 is an array containing OLI-derived Rrs pixels
#y_vals_train1 is an array containing MSI-derived Rrs pixels
x_vals_train1 = np.array(file1['MSI-Band'])   # input Feature (Rrs-MSI)
y_vals_train1 = np.array(file1['OLI-Band'])   # Labels Rrs_OLI


x_batch=  np.reshape(x_vals_train1, [len(x_vals_train1),1])
y_batch=  np.reshape(y_vals_train1, [len(y_vals_train1),1])



######### Model Initialization#######
tf.set_random_seed(1)
np.random.seed(1)
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder_with_default(False, (), 'is_training')
init = tf.global_variables_initializer()   
         
                              
# No of neurons and Hidden Layers, Batch size, learning rate    
neurons1=25
Epoch=1000
batch_size= 32
lr=0.0075
momentum=0.5

neurons2=25 
neurons3=25
neurons4 = 25

       
def neural_net_model(X_data,input_dim):
                    #W_1 = tf.Variable(tf.random_uniform([input_dim,100]))
                    W_1 = tf.Variable(initializer([input_dim,neurons1]))                                # this one
                    #W_1=tf.Variable(tf.truncated_normal([input_dim,neurons1],mean = 0.0,stddev=0.1))
                    #b_1 = tf.Variable(tf.zeros([250]))
                    b_1 = tf.Variable(initializer([neurons1])) # this one
                    #b_1=tf.Variable(tf.constant(0.1,shape = [neurons1]))
                    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
                    layer_1 = tf.nn. tanh(layer_1)
                    #layer_1= tf.layers.batch_normalization(layer_1)
                
                    layer_1=  tf.layers.batch_normalization(layer_1, training=is_training)
                    layer_1  = tf.nn.dropout( layer_1 , keep_prob)
                    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    #with tf.control_dependencies(update_ops):
                    layer_1 = tf.identity(layer_1)
                
        
                    #W_2 = tf.Variable(tf.random_uniform([100,1000]))
                    W_2 = tf.Variable(initializer([neurons1,neurons2]))   # this one
                    #W_2=tf.Variable(tf.truncated_normal([neurons1,neurons2],mean = 0.0,stddev=0.1))
                    #b_2 = tf.Variable(tf.zeros([1000]))
                    b_2 = tf.Variable(initializer([neurons2])) # this one
                    #b_2=tf.Variable(tf.constant(0.1,shape = [neurons2]))
                    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
                    layer_2 = tf.nn. tanh(layer_2)
                    #layer_2= tf.layers.batch_normalization(layer_2)
                    layer_2=  tf.layers.batch_normalization(layer_2, training=is_training)
                    layer_2  = tf.nn.dropout( layer_2 , keep_prob)
                    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    #with tf.control_dependencies(update_ops):
                    layer_2 = tf.identity(layer_2)
        #            
                    #W_3 = tf.Variable(tf.random_uniform([1000,1000]))
                    W_3 = tf.Variable(initializer([neurons2,neurons3])) # this one
                    #b_3 = tf.Variable(tf.zeros([1000]))
                    b_3 = tf.Variable(initializer([neurons3]))  # this one
                    
                    layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
                    #layer_3 = tf.nn.leaky_ tanh(layer_3)
                    layer_3 = tf.nn. tanh(layer_3)
                    #layer_3= tf.layers.batch_normalization(layer_3)
                
                    layer_3=  tf.layers.batch_normalization(layer_3, training=is_training)
                    layer_3  = tf.nn.dropout( layer_3 , keep_prob)
                    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    #with tf.control_dependencies(update_ops):
                    layer_3 = tf.identity(layer_3)
                    
                    
                    #W_O = tf.Variable(tf.random_uniform([1000,1]))
                    W_O = tf.Variable(initializer([neurons3,1]),name='W_O')
                    #W_O=tf.Variable(tf.truncated_normal([neurons,1],mean = 0.0,stddev=0.1))
                    #b_O = tf.Variable(tf.zeros([1]))
                    b_O = tf.Variable(initializer([1]),name='b_O')
                    #b_O=tf.Variable(tf.constant(0.1,shape = [1]))
                    output = tf.add(tf.matmul(layer_3,W_O), b_O)
                    return output,W_O,b_O           
#                    


############ Normalizing if needed ###########
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

####### Feature extension#######3
x_batch= np.concatenate((x_batch, scaler.fit_transform(x_batch), np.log10(x_batch)),axis=1)
y_batch = np.log10(y_batch)




x_data = tf.placeholder(tf.float32, [None,x_batch.shape[1]], name = 'x')
keep_prob = tf.placeholder(tf.float32)
y_target = tf.placeholder(tf.float32, [None, 1], name = 'y')  # number of outputs  == number of columns
L_data = tf.placeholder(tf.float32, [1], name = 'z')  # number of features == number of columns
n_d=len(x_batch[0])

    
initializer = tf.contrib.layers.xavier_initializer(uniform=False,seed=10)
model_output,W_O,b_O= neural_net_model(x_data,n_d)
epsilon = tf.constant([0.5])
global_step = tf.Variable(0, trainable=False)
learning_rate =  tf.train.exponential_decay(lr, global_step,100, 0.95, staircase=True) 


reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_target, model_output)))) 
loss = tf.add_n([loss] + reg_losses, name="loss")



############Choice of using different optimizers#################

#my_opt = tf.train.GradientDescentOptimizer(lr)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
my_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)#,beta1=0.9, beta2=0.999, epsilon=1e-10)
#my_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.9,epsilon=1e-10)
#my_opt = tf.train.RMSPropOptimizer(learning_rate=0.0001)
# clippig gradient
#my_opt = tf.contrib.estimator.clip_gradients_by_norm(my_opt, clip_norm=1.0)
#my_opt = tf.train.AdagradOptimizer(learning_rate=lr)
#my_opt=tf.train.MomentumOptimizer(learning_rate, momentum)
#with tf.control_dependencies(update_ops):


saver = tf.train.Saver()   
train_step = my_opt.minimize(loss,global_step)


########  5-fold experemnts ##########3       
from sklearn.model_selection import KFold 
#from sklearn.model_selection import RepeatedKFold
#kf = KFold(len(x_batch),True,1) 
kf = KFold(5,random_state=None,shuffle=True) # 5,random_state=14,shuffle=True
for train_index, test_index in kf.split(x_batch):
        x_vals_train1, x_vals_test = x_batch[train_index], x_batch[test_index] 
        y_vals_train1, y_vals_test = y_batch[train_index], y_batch[test_index]
        
        train_indices = np.random.choice(len(x_vals_train1), round(len(x_vals_train1)*0.70), replace=False)
        test_indices = np.array(list(set(range(len(x_vals_train1))) - set(train_indices)))
        x_vals_train = x_vals_train1[train_indices]
        x_vals_val = x_vals_train1[test_indices]
        y_vals_train = y_vals_train1[train_indices]
        y_vals_val = y_vals_train1[test_indices]
        
        x_vals_train1=np.concatenate((x_vals_train,x_vals_val),axis=0)
        y_vals_train1=np.concatenate((y_vals_train,y_vals_val),axis=0)
            

 
        train_loss = []
        test_loss =[]
        best_R2=-1000000
        Best_R2_val=-10000
        
        test_R2=[]
        train_R2=[]
        val_R2=[]
        print("starting Training the Model ")
    
        init = tf.global_variables_initializer()
        train_step=tf.group([train_step,update_ops])
        
        
        dropout_prob = 1
        eval_Loss=[]
        val_loss=[]
        
        best_rmse_val=25
        best_mape_val= 1000
        best_rmse_train=25 
        best_med = 25
        E=100
        R2_Test=[]
        RMSE_Test =[]
        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.global_variables_initializer())    
            
            print("startingTraining1")
            for i in range(100):
        #       
                # Batch Normalizatin##############################################3##
                rand_index = np.random.choice(len(x_vals_train), size=batch_size)
                X = np.transpose([x_vals_train[rand_index]])
                X = x_vals_train[rand_index]
                Y = np.transpose([y_vals_train[rand_index]])
                #####################################################################
                s=Y.shape[1]
                Y= np.reshape(Y, [s,1])
                sess.run(train_step, feed_dict={x_data: X, y_target: Y,keep_prob:dropout_prob,is_training: True})
                
                temp_train_loss = sess.run(loss, feed_dict={x_data:x_vals_train, y_target:y_vals_train,keep_prob:dropout_prob,is_training: True})
                train_loss.append(temp_train_loss)
                temp_val_loss = sess.run(loss, feed_dict={x_data:x_vals_val, y_target:y_vals_val,keep_prob:1,is_training: False})
                val_loss.append(temp_val_loss)
                
                
                temp_train_R2 =r2_score(y_vals_train, sess.run( model_output, feed_dict={x_data:x_vals_train,keep_prob:dropout_prob,is_training: True}))
                train_R2.append(temp_train_R2)
                temp_val_R2 = r2_score(y_vals_val, sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: False}))
                temp_val_mape= 100*mean_absolute_percentage_error(y_vals_val, sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: False}))
                val_R2.append(temp_val_R2)
                MD_train= abs(np.median(y_vals_val - sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: True})))
                MD_val= np.median(y_vals_val - sess.run( model_output, feed_dict={x_data:x_vals_val,keep_prob:1,is_training: False}))

                MD_val= abs (MD_val)
                
                ####### Early Stopping ########
                if best_rmse_val > temp_val_loss: # and best_mape_val > temp_val_mape: # and best_R2 < temp_val_R2) and  
                   best_rmse_val =temp_val_loss
                   best_rmse_train= temp_train_loss
                   best_R2 = temp_val_R2
                   best_med = MD_val
                   best_mape_val  = temp_val_mape
                   best_epoch=i

                   y_pred_batch = sess.run(model_output, feed_dict={x_data:x_vals_test, y_target: y_vals_test,keep_prob:1,is_training: False})
                   
                
                   rmse_T= sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_vals_test, y_pred_batch)))))
                   R2_T=   r2_score(y_vals_test, sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))  
                   MD_T= np.median(y_vals_test - sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))
                   mape_T= 100*mean_absolute_percentage_error(y_vals_test, sess.run( model_output, feed_dict={x_data:x_vals_test,keep_prob:1,is_training: False}))

                      
                if (i % 1000==0):   
#                           print(i, "Trian Loss:",temp_train_loss,"Val_Loss=", temp_val_loss, "Test Loss:",temp_test_loss, "Best_val_Loss:", best_rmse_val)
#                           print(i,'R2score_train:',temp_train_R2,'R2score_val:',temp_val_R2, 'R2score_test:',temp_test_R2, 'Best_R2_val=',best_R2)  
                    
                   print(i, "Trian Loss:",temp_train_loss,"Val_Loss=", temp_val_loss, "Best_val_Loss:", best_rmse_val, "Test Loss:", (rmse_T))
                   print(i, "temp_val_mape=:", temp_val_mape,  "Test_mape:", mape_T)
                   print(i,'R2score_train:',temp_train_R2,'R2score_val:',temp_val_R2, 'R2score_test:',R2_T, 'Best_R2_val=',best_R2)
                   
                   
       
                      
            
            sess.close()    
            R2_Test_All.append(R2_Test)       
            
            Pre_Data = np.append (Pre_Data,y_pred_batch ) #Pre_Data = np.append (Pre_Data,y_pred_batch )
            OLI_Data = np.append (OLI_Data,y_vals_test )  # This data is to check the model performamce 
            
            # v= x_vals_test
            # v= np.reshape(v, [len(v),3])
            # MSI_Data= np.append (MSI_Data, v )    # Ref
            
            msi_ref = x_vals_test[:, 0]   # raw MSI band used for plotting/comparison
            MSI_Data = np.append(MSI_Data, msi_ref)
            #reg = linear_model.LinearRegression().fit(np.reshape(x_vals_train1[:,2], [len(x_vals_train1[:,2]),1]), y_vals_train1)
            reg = linear_model.LinearRegression().fit(np.reshape(x_vals_train1, [len(x_vals_train1),3]), y_vals_train1)
            #lin= reg.predict(np.reshape(x_vals_test[:,2], [len(x_vals_test[:,2]),1]))
            lin= reg.predict(np.reshape(x_vals_test, [len(x_vals_test),3]))
            Linear_P=np.append(Linear_P, lin)
            print('len of MSI Data.............', len(MSI_Data), len(x_vals_test))
 

MSI_Data=10**(MSI_Data)   

Pre_Data=10**(Pre_Data)  

#GP_Data= 10**(GP)
OLI_Data=10**(OLI_Data)
Linear_P= 10**(Linear_P)

MSI_Data= MSI_Data * np.pi
Linear_P= np.pi*(Linear_P)
OLI_Data=np.pi*(OLI_Data)
Pre_Data=np.pi*(Pre_Data)  



    ###################################################################################
### NO Band Adjustment 
plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.85)
xdata =  OLI_Data 
ydata =  MSI_Data 

indy= np.where(ydata == np.amax(ydata))
ydata= np.delete(ydata, indy)
xdata= np.delete(xdata, indy)

indx= np.where(xdata == np.amax(xdata))
xdata= np.delete(xdata, indx)
ydata= np.delete(ydata, indx)


xdata=np.reshape(xdata, [len(xdata),1])
ydata=np.reshape(ydata, [len(ydata),1])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel( '$\u03C1_{w}$'+  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
ax.set_ylabel('$\u03C1_{w}$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2',fontsize=30)
xlim=0
ylim=0.05
ax.set_ylim(xlim, ylim)
ax.set_xlim(xlim, ylim) 
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
plt.legend(fontsize=24)
ax.set_title('No-Bandpass Adjustment',fontsize=30)
for axis in ['top','bottom','left','right']:
 ax.spines[axis].set_linewidth(4)
plt.savefig('No BA-'+'Band' + MSI_Band[0:7], bbox_inches = 'tight', dpi=600)
plt.show()


MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
rmse=mean_squared_error(ydata, xdata, squared=False)
BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
MD= np.median(xdata - ydata)
z=np.median((np.log(ydata/xdata)))
z= 100 * np.sign(z) * (10**abs(z) -1)

y=np.median(abs((np.log(ydata/xdata))))
y=100 * (10**y -1)

r2= r2_score(xdata,ydata)
rRMSD= np.sqrt(np.mean(np.square((ydata-xdata)/xdata)))*100

corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
r= corr1 * corr1
slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))


print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
print('Results for Band' + MSI_Band[0:7], '\n', file=f)
print('Results before Band Conversion:','\n', file=f)
print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)  
  
plt.figure()
fig, ax = plt.subplots()
kwargs = dict(alpha=0.5, bins=50)
plt.hist(MSI_Data, **kwargs, color='orange', label='MSI')
plt.hist(OLI_Data, **kwargs, color='dodgerblue', label='OLI')
plt.gca().set(title='Frequency Histogram of OLI-MSI')
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
plt.xlim(0.0001,ylim)
plt.legend();
plt.savefig('xxorg Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)

plt.figure(figsize=(10,7), dpi= 80)
fig, ax = plt.subplots()
kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':4})
sns.distplot(MSI_Data, color="orange", label="MSI", **kwargs)
sns.distplot(OLI_Data, color="dodgerblue", label="OLI", **kwargs)

ax.tick_params(axis='both', which='major', width=10, labelsize=30)
plt.xlim(0.000,ylim)
plt.legend();
plt.savefig('xxorg Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)


#plt.legend();
#############################3
################ NN ####################
    
plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.85)
xdata =  (OLI_Data)
ydata =  (Pre_Data)


indy= np.where(ydata == np.amax(ydata))
ydata= np.delete(ydata, indy)
xdata= np.delete(xdata, indy)

indx= np.where(xdata == np.amax(xdata))
xdata= np.delete(xdata, indx)
ydata= np.delete(ydata, indx)

xdata=np.reshape(xdata, [len(xdata),1])
ydata=np.reshape(ydata, [len(ydata),1])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


ax.set_xlabel( '$\u03C1_{w}$'+  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
ax.set_ylabel('$\u03C1_{w}$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2' + '*',fontsize=30)


#ax.plot([],[],' ',color = "blue", label='OCN')
ax.set_title('NN Bandpass Adjustment',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
ax.set_ylim(xlim, ylim)
ax.set_xlim(xlim, ylim) 
#ax.set_xticks(xlim, ylim,0.04)

ax.tick_params(axis='both', which='major', width=10, labelsize=30)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(4)
plt.legend(fontsize=24)

plt.savefig('Conversion-MSI-OLI-8' + 'Band' + str(MSI_Band[0:7]), bbox_inches = 'tight', dpi=600)
plt.show()

MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
rmse=mean_squared_error(ydata, xdata, squared=False)
BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
MD= np.median(xdata - ydata)
z=np.median((np.log(ydata/xdata)))
z= 100 * np.sign(z) * (10**abs(z) -1)

y=np.median(abs((np.log(ydata/xdata))))
y=100 * (10**y -1)

r2= r2_score(xdata,ydata)

corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
r= corr1 * corr1

slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))

slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(ydata,-1),np.reshape(xdata,-1))

print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
print('Results for Band' + MSI_Band[0:7], '\n', file=f)
print('Results after NN Band Conversion:','\n', file=f)
print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
 
print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)  
   

plt.figure()
fig, ax = plt.subplots()
kwargs = dict(alpha=1, bins=10)
plt.hist(xdata, **kwargs, color='orange', label='OLI')
plt.hist(ydata, **kwargs, color='dodgerblue', label='MSI')
plt.gca().set(title='Frequency Histogram of OLI-MSI'+'*')
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
plt.xlim(0.0001,ylim)
plt.savefig('NN Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)


plt.figure(figsize=(10,7), dpi= 80)
fig, ax = plt.subplots()
kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':4})
sns.distplot(xdata, color="dodgerblue", label="OLI", **kwargs)
sns.distplot(ydata, color="orange", label="MSI", **kwargs)
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
plt.xlim(0.0001,ylim)
#plt.legend();
plt.savefig('NN Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
    
     
    ##############################################3333   
    
 

    
##################################################################################3    
xdata= OLI_Data
ydata= Linear_P

MAPD= 100* np.median(abs((np.array(ydata)-np.array(xdata))/np.array(xdata)))
MRPD= 100* np.median((np.array(ydata)-np.array(xdata))/np.array(xdata))
rmsle=mean_squared_error(np.log10(ydata), np.log10(xdata), squared=False)
rmse=mean_squared_error(ydata, xdata, squared=False)
BIAS=  10**  1/ len(xdata)*(np.median(np.log(ydata)-np.log10(xdata)))
MD= np.median(xdata - ydata)
z=np.median((np.log(ydata/xdata)))
z= 100 * np.sign(z) * (10**abs(z) -1)

y=np.median(abs((np.log(ydata/xdata))))
y=100 * (10**y -1)

r2= r2_score(xdata,ydata)

corr1, p_value1 = pearsonr(np.reshape(xdata,-1), np.reshape(ydata,-1))
r= corr1 * corr1
slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(xdata,-1),np.reshape(ydata,-1))

print('slope', slope, 'intercept' , intercept, 'MD', MD, "MRPD", MRPD, "MAPD", MAPD, "RMSLE", rmsle, 'RMSE', rmse, "R2", r2,  'symmetric signed percentage bias ', z , 'median symmetric accuracy', y) 
f = open('Results_' + MSI_Band[0:7] + '.txt' , 'a')
print('Results for Band' + MSI_Band[0:7], '\n', file=f)
print('Results after Liner Band Conversion:','\n', file=f)
print('slope', slope, 'intercept' , intercept, 'MD', MD, '\n', "MRPD", MRPD,  '\n',  "MAPD", MAPD, '\n',"RMSLE", rmsle, '\n', 'RMSE', rmse, '\n',"R2", r2,  '\n', 'symmetric signed percentage bias ', z ,'\n', 'median symmetric accuracy', y,'\n', file=f)
print("{:e}".format(round(rmse,6)), '&', round(rmsle,3) ,'&', round(MD,6), "&", round(MAPD,3), "&", round(MRPD,3), "&", round(r2,3),  '&', round(slope,3), '&' ,  round(intercept,6),file=f)
  
  

plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.85)
xdata =  (OLI_Data)
ydata =  (Linear_P)
indy= np.where(ydata == np.amax(ydata))
ydata= np.delete(ydata, indy)
xdata= np.delete(xdata, indy)

indx= np.where(xdata == np.amax(xdata))
xdata= np.delete(xdata, indx)
ydata= np.delete(ydata, indx)
xdata=np.reshape(xdata, [len(xdata),1])
ydata=np.reshape(ydata, [len(ydata),1])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(xdata, ydata,  marker='o', s=150) #, c= ydata, cmap='Spectral')
line = mlines.Line2D([0, 1], [0, 1], color='black',label='1:1 Line',linewidth=4.0 )
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax.set_xlabel( '$\u03C1_{w}$'+  ' '+'[' +OLI_Band[4:7]+'nm]'+ '-'+ 'OLI-L8',fontsize=30)
ax.set_ylabel('$\u03C1_{w}$' +  ' '+'[' +MSI_Band[4:7]+'nm]'+ '-'+ 'MSI-S2' + '*',fontsize=30)


#ax.plot([],[],' ',color = "blue", label='OCN')
ax.set_title('OLS Bandpass Adjustment',fontsize=30)
#    min1= np.min (xdata)
#    min2= np.min (ydata)
ax.set_ylim(xlim, ylim)
ax.set_xlim(xlim, ylim) 
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
for axis in ['top','bottom','left','right']:
 ax.spines[axis].set_linewidth(4)
plt.legend(fontsize=24)
plt.savefig('Linear Conversion-MSI-' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
plt.show()

import seaborn as sns
plt.figure()
plt.figure(figsize=(10,7), dpi= 80)
fig, ax = plt.subplots()
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':4})
sns.distplot(xdata, color="dodgerblue", label="OLI", **kwargs)
sns.distplot(ydata, color="orange", label="MSI", **kwargs)
ax.tick_params(axis='both', which='major', width=10, labelsize=30)
plt.xlim(0.0001,ylim)
#plt.legend();
plt.savefig('Linear Hist' + 'Band' + str(OLI_Band[0:7]), bbox_inches = 'tight', dpi=600)
