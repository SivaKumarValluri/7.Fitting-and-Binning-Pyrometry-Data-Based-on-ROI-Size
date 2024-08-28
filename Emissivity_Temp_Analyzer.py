# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:33:05 2024

@author: Siva Kumar Valluri
"""

import os
import glob
import io 
import numpy as np
import math
#from scipy.interpolate import interp1d
#from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as pcip
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import statistics as stat
from scipy.signal import savgol_filter

#functions used:---------------------------------------------------------------------------------------------------------------------------------------------   

def fit_derivative_and_integral(x,y):
    #f = interp1d(x, y, kind='cubic')
    #f = CubicSpline(x,y)
    f=pcip(x, y, axis=0, extrapolate=None)
    #f = UnivariateSpline(x,y)
    x_new= np.logspace(math.log10(10), math.log10(90000), num=1000, endpoint=True,base=10.0)
    logx_new=[math.log10(x_new[i]) for i in range(len(x_new))]
    y_new=f(x_new)  
    dydx=np.gradient(y_new,x_new)
    dydlogx=np.gradient(y_new,logx_new)
    cd=integrate.cumtrapz(y_new, x_new)
    cdf=np.array([0])
    cdf=np.append(cdf,cd)     
    return [x_new,y_new,dydx,dydlogx,cdf]

def fit_2(x,y):
    #f = interp1d(x, y, kind='cubic')
    #f = CubicSpline(x,y)
    xstart = x[0]
    xend = x[-1]
    f=pcip(x, y, axis=0, extrapolate=None)
    #f = UnivariateSpline(x,y)
    x_new= np.arange(xstart,xend,1)
    y_new=f(x_new)     
    return [y_new]

def extrema_finder(x,y):
    pks, _ = find_peaks(y,height=0)
    #extreme points
    xx=[x[pks[i]] for i in range(len(pks))]
    yy=[y[pks[i]] for i in range(len(pks))]
    return [xx,yy]


def plotmepls(xr,yr,x2,y2,x_ex,y_ex,x):
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('time,ns')
            ax1.title.set_text('Radiance and dR/dlogt plots along with extrema in run '+str(x))
            plt.xscale('log')
            ax1.set_ylabel('Radiance, W/Sr.m^2', color = 'black') 
            ax1.plot(xr, yr, color = 'black')
            ax1.plot(xr,np.zeros_like(xr), linestyle = 'dotted', color = 'red' )
            plt.scatter(x_ex,y_ex,s=25,marker = 'o',color = 'black')
            ax2 = ax1.twinx()
            ax2.set_ylabel('dR/dlogt, arb', color = 'blue') 
            ax2.plot(x2, y2, color = 'blue')
            ax2.plot(x2,np.zeros_like(x2), linestyle = 'dotted', color = 'red' )
            ax2.tick_params(axis ='y', labelcolor = 'blue')
            #ax2.set_yscale('log')
            return plt.show()

def plotmepls2(x1,y1,x2,y2,Tx_measured,Ty_measured,Ex_measured,Ey_measured,x):
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('time,ns')
            ax1.title.set_text('Temperature and emissivity in run '+str(x))
            plt.xscale('log')
            ax1.set_ylabel('Tempeprature, K', color = 'black') 
            ax1.plot(x1, y1, color = 'black')
            ax1.plot(x1,np.ones_like(x1)*1500, linestyle = 'dotted', color = 'red' )
            plt.scatter(Tx_measured,Ty_measured,s=15,marker = 'o',color = 'black')
            ax2 = ax1.twinx()
            ax2.set_ylabel('emissivity, arb', color = 'blue') 
            ax2.plot(x2, y2, color = 'blue')
            ax2.plot(x2,np.zeros_like(x2), linestyle = 'dotted', color = 'red' )
            ax2.plot(x2,np.ones_like(x2)*0.05, linestyle = 'dotted', color = 'red' )
            plt.scatter(Ex_measured,Ey_measured,s=15,marker = 'o',color = 'blue')
            ax2.tick_params(axis ='y', labelcolor = 'blue')
            return plt.show()

def plotmepls3(x1,a,x2,b,x3,c,name):
            from sklearn import preprocessing
            y1 = np.transpose(preprocessing.normalize([np.array(a)])) #normalized radiance
            #y2 = np.transpose(preprocessing.normalize([np.array(b)]))#normalized temperature 
            y3 = np.transpose(preprocessing.normalize([np.array(c)])) #normalized emissivity
            y2 = b                                                    #temperature in K 
            marker = 0.10*float(y3.max())                             #10 % of peak intensity 
            fig, ax1 =plt.subplots()
            ax1.set_xlabel('time,ns')
            ax1.title.set_text('Normalized dataset of '+str(name))
            plt.xscale('log')
            ax1.set_ylabel('Normalized depedent variable, arb', color = 'black') 
            ax1.plot(x1, y1, color = 'blue', label = 'radiance')
            plt.scatter(x1,y1,s=15,marker = 'o',color = 'blue')
            ax1.plot(x3, y3, color = 'green', label = 'emissivity')
            plt.scatter(x3,y3,s=10,marker = 'o',color = 'green')
            ax1.plot(x1, np.zeros_like(x1), linestyle = 'dotted', color = 'black' )
            ax1.plot(x1, marker*np.ones_like(x1), linestyle = 'dotted', color = 'black' )
            ax2 = ax1.twinx()
            ax1.grid(which='minor', color='orange', linestyle=':') 
            ax2.set_ylabel('temperature, K', color = 'red')
            ax2.plot(x2, y2, color = 'red', label = 'temperature')
            ax2.tick_params(axis ='y', labelcolor = 'red')
            plt.scatter(x2,y2,s=10,marker = 'o',color = 'red')
            ax2.plot(x1, 4000*np.ones_like(x1), linestyle = 'dotted', color = 'red' )
            ax2.plot(x1, 3000*np.ones_like(x1), linestyle = 'dotted', color = 'red' )
            ax2.plot(x1, 2000*np.ones_like(x1), linestyle = 'dotted', color = 'red' )
            ax1.legend()
            plt.gcf().set_dpi(500)
            return plt.show()

        
        



def main():
    address = input("Enter address of folder with binarized images (just copy paste address): ")
    txt_files = glob.glob(os.path.join(address, "*radiance.txt"))
    txt_files_2 = glob.glob(os.path.join(address, "*grayTemp.txt"))
    txt_files_3 = glob.glob(os.path.join(address, "*grayPhi.txt"))
    folder_name=address.rpartition('\\')[0].rpartition('\\')[2] #folder name is assumed sample/condition detail
    
    #Excelwriter = pd.ExcelWriter(str(folder_name)+'.xlsx')

    name_list =[]
    for t in range(len(txt_files)):
        #reading txt file: radiance
        temps=[]
        with io.open(txt_files[t], mode="r") as f:    
            next(f) #label
            next(f) #units
            next(f) #file name
            #copying data
            for line in f:
                temps.append(line.split())
        
        temp=np.array(temps, dtype=np.float32) #actual temporary file 
        temps=[]
        
        #reading txt file for names
        names=[]
        with io.open(txt_files[t], mode="r") as f:    
            next(f) #label
            next(f) #units
            #copying data
            for line in f:
                names.append(line.split())
        
        name=pd.DataFrame(names[0]).drop_duplicates() #actual temporary file 
        name = name.values.tolist()
        names=[]
    
        #reading txt file: corresponding temperature data  
        temps_2=[]
        with io.open(txt_files_2[t], mode="r") as f:    
            next(f) #label
            next(f) #units
            next(f) #file name
            #copying data
            for line in f:
                temps_2.append(line.split())
        
        temp_2=np.array(temps_2, dtype=np.float32) #actual temporary file 
        temps_2=[]
    
        #reading txt file: corresponding emissivity data  
        temps_3=[]
        with io.open(txt_files_3[t], mode="r") as f:    
            next(f) #label
            next(f) #units
            next(f) #file name
            #copying data
            for line in f:
                temps_3.append(line.split())
        
        temp_3=np.array(temps_3, dtype=np.float32) #actual temporary file 
        temps_3=[]
        
        file_analyzed = txt_files[t].split('\\')[-1]
        file_analyzed = file_analyzed.split('.txt')[0]
        
        #pocessing each sample: has several runs in txt file
        Tlist=[]
        Elist=[]
        Dmain = pd.DataFrame(columns = ['start/ns', 'end/ns','Max emissivity', 'time maxE/ns', 'Max T/K', 'time maxT/ns', 'Av Comb/Deflag T/K', 'stdev T/K'])
        for i in range(0,np.shape(temp)[1],2):
            
            #Radiance 
            t_1=[temp[j,i] for j in range(len(temp[:,i]))] 
            R=[temp[j,i+1] for j in range(0,len(t_1),1)] 
            dfR= pd.DataFrame(columns = ['time','radiance'])
            dfR['time'] = pd.Series(t_1)
            dfR['radiance'] = pd.Series(R)
            dfR = dfR.drop_duplicates(subset=['time'], keep='last') # removing redundant time entries necessary for fitting
            dfR = dfR.drop(dfR.index[dfR['time'] < 0]) # removing negative time entries
            dfR = dfR.dropna(subset=['radiance'])
            
            #Temperature
            corr=int(i*(3/2)) 
            dfT= pd.DataFrame(columns = ['time','temperature','error'])
            t_2=[temp_2[j,corr] for j in range(len(temp_2[:,corr]))]
            dfT['time'] = pd.Series(t_2) 
            dfT['temperature'] = pd.Series([temp_2[j,corr+1] for j in range(len(t_2))]) 
            dfT['error'] = pd.Series([temp_2[j,corr+2] for j in range(len(t_2))]) 
            dfT = dfT.drop_duplicates(subset=['time'], keep='last') # removing redundant time entries necessary for fitting
            dfT = dfT.drop(dfT.index[dfT['time'] < 0]) # removing negative time entries
            dfT = dfT.dropna(subset=['temperature','error'])
            
            #Emissivity
            corr=int(i*(3/2))
            dfE= pd.DataFrame(columns = ['time','E','error'])
            t_3=[temp_3[j,corr] for j in range(len(temp_3[:,corr]))] 
            dfE['time'] = pd.Series(t_3) 
            dfE['E'] = pd.Series([temp_3[j,corr+1] for j in range(len(t_3))])
            dfE['error'] = pd.Series([temp_3[j,corr+2] for j in range(len(t_3))]) 
            dfE = dfE.drop_duplicates(subset=['time'], keep='last') # removing redundant time entries necessary for fitting
            dfE = dfE.drop(dfE.index[dfE['time'] < 0]) # removing negative time entries
            dfE = dfE.dropna(subset=['E','error'])
            
            #Viewing data
            plotmepls3(dfR['time'], dfR['radiance'], dfT['time'], dfT['temperature'], dfE['time'], dfE['E'],"file no. " +str(name[int(i/2)][0]))

            while True:
                skip_choice = input("Do you want to process this one ? Yes/No : ").lower() or "yes" # This will make the user input not case-senitive
                try:
                    if skip_choice.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
                        #Fitting radiance data and arriving at instantaneous differential and cumulative integral values
                        [Rx_new,Ry_new,dRydx,dRydlogx,Rcdf] = fit_derivative_and_integral(np.array(dfR['time']),np.array(dfR['radiance']))
                        [x_ex, y_ex] = extrema_finder(Rx_new,Ry_new)
                        #plotmepls(Rx_new,Ry_new,Rx_new,dRydlogx,x_ex,y_ex,"Run number " +str(name[int(i/2)][0]))
                        
                        #Fitting temperature and emissivity data
                        [Tx_new,Ty_new,_,_,_] = fit_derivative_and_integral(np.array(dfT['time']),np.array(dfT['temperature']))
                        [Ex_new,Ey_new,_,_,Ecdf] = fit_derivative_and_integral(np.array(dfE['time']),np.array(dfE['E']))
                        #plotmepls2(Tx_new,Ty_new,Ex_new,Ey_new,np.array(dfT['time']),np.array(dfT['temperature']),np.array(dfE['time']),np.array(dfE['E']),"Run number " +str(name[int(i/2)][0]))
                        
                        
                        #Extreme point/ local peak finder
                        Exx,Eyy = extrema_finder(Ex_new,Ey_new)
                        Txx,Tyy = extrema_finder(Tx_new,Ty_new)

                        print("Check the plots to provide subsequent answers")
                                
                        #Manual zone definition
                        point1 = int(input("Enter lower limit time in ns for identifying combustion/deflag: "))
                        point2 = int(input("Enter upper limit time in ns for identifying combustion/deflag: "))
                        
                        Exx_1=[x for x in Exx if point1<x<point2] 
                        indexE = [Exx.index(x) for x in Exx_1]
                        Eyy_1 = [Eyy[x] for x in indexE]
                        E_max = max(Eyy_1)       
                        time_max =  Exx[Eyy.index(E_max)]
                        
                        Tx = [x if point1<= x <= point2 else 0 for x in Tx_new]
                        Tx = [x for x in Tx if x != 0]
                        indexT = [Tx_new.tolist().index(x) for x in Tx]
                        Ty = [Ty_new[x] for x in indexT]
                        Tlist.append(Ty)
                        Tyfit  = fit_2(Tx,Ty)
                        Tmean = np.mean(Tyfit[0])
                        Tstd = np.std(Tyfit[0])
                        
                        Ex = [x if point1<= x <= point2 else 0 for x in Ex_new]
                        Ex = [x for x in Tx if x != 0]
                        indexE2 = [Ex_new.tolist().index(x) for x in Ex]
                        Ey = [Ey_new[x] for x in indexE2]
                        Elist.append(Ey)
                        
                        point111 = int(input("Enter lower limit time for initial peak identification in ns: "))
                        point222 = int(input("Enter upper limit time for initial peak identification in ns: "))
                        
                        Txx_1=[x for x in Txx if point111<x<point222] 
                        indexp1 = [Txx.index(x) for x in Txx_1]
                        Tyy_1 = [Tyy[x] for x in indexp1]
                        T_max = max(Tyy_1)
                        time_max2 =  Txx[Tyy.index(T_max)]
                        
                        print("For run number " +str(name[int(i/2)][0]))
                        print(str('%.2f' % E_max) +' maximum emissivity measured at ' + str('%.2f' % time_max) + ' ns')   
                        print(str('%.2f' % T_max) +' K measured at ' + str('%.2f' % time_max2) +' ns')
                        
                        name_list.append(str(name[int(i/2)][0]))
                        AllXY = np.column_stack((point1, point2, E_max,time_max,T_max, time_max2, Tmean, Tstd))
                        X = pd.DataFrame(AllXY,columns = ['start/ns', 'end/ns','Max emissivity', 'time maxE/ns', 'Max T/K', 'time maxT/ns', 'Av Comb/Deflag T/K', 'stdev T/K'])
                        Dmain = Dmain._append(X, ignore_index = True)
                        
                    elif skip_choice.lower() in ["n","no","nope"]:
                        print("Good for you, you escapist")
                        name_list.append(str(name[int(i/2)][0]))
                        AllXY = np.column_stack((0, 0, 0, 0, 0, 0, 0, 0))
                        X = pd.DataFrame(AllXY,columns = ['start/ns', 'end/ns','Max emissivity', 'time maxE/ns', 'Max T/K', 'time maxT/ns', 'Av Comb/Deflag T/K', 'stdev T/K'])
                        Dmain = Dmain._append(X, ignore_index = True)
                    else:
                        raise Exception("Invalid input! Answer can only be 'Yes' or 'No'")
                except Exception as e:
                    print(e)    
                else:
                    break                        

            
            while True:
                continue_choice = input("Do you want to continue with remaining runs? Yes/No : ").lower() or "yes" # This will make the user input not case-senitive
                try:
                    if continue_choice.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
                        print("You've gone through " + str(int(name.index(name[int(i/2)]))+1) + " out of " + str(len(name)) + " runs")
                    elif continue_choice.lower() in ["n","no","nope"]:
                        print("Good for you, you escapist")
                        return folder_name, name_list, Tlist, Elist, Dmain
                    else:
                        raise Exception("Invalid input! Answer can only be 'Yes' or 'No'")
                except Exception as e:
                    print(e)    
                else:
                    break
            
    return folder_name, name_list, Tlist, Elist, Dmain

#main-------------------------------------------------------------------------------------------------------------------------------------------------------

folder_name, name_list, Tlist, Elist, Dmain = main()
Tframe = pd.DataFrame(Tlist)
Eframe = pd.DataFrame(Elist)
#Tframe.set_index(name_list)
#Dmain.set_index(name_list)
"""
Tframe.to_excel(str(folder_name)+"-comb-deflag Temp output1.xlsx") 
Eframe.to_excel(str(folder_name)+"-comb-deflag output.xlsx") 
Dmain.to_excel(str(folder_name)+"-extrema output.xlsx")
"""
Tframe.to_excel("Al-CuO-comb-deflag Temperature output.xlsx") 
Eframe.to_excel("Al-CuO-comb-deflag Emissivity output.xlsx") 
Dmain.to_excel("Al-CuO-extrema output.xlsx")        

        

        