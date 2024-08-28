# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:19:40 2024

@author: Siva Kumar Valluri
"""

class Prometheus():
    run_plot_choice = 'y'
    iterative_plot_choice = 'n'
    error_confidence = 'n'
    binned_plots_choice = 'n'
    def __init__(self):
        import os
        import glob
        import io 
        import numpy as np
        import pandas as pd
        
        self.address = input("Enter address of folder with PMT txt files (just copy paste address): ")
        self.sample_name = input("What do you want to call this sample? : ")
        
        radiance_txt_files = glob.glob(os.path.join(self.address, "*radiance.txt"))
        temperature_txt_files = glob.glob(os.path.join(self.address, "*grayTemp.txt"))
        emissivity_txt_files = glob.glob(os.path.join(self.address, "*grayPhi.txt"))
        
        #reading txt file for names
        names = []
        with io.open(radiance_txt_files[0], mode="r") as f:    
                next(f) #label
                next(f) #units
                #copying data
                for line in f:
                    names.append(line.split())
                        
        names_list = pd.DataFrame(names[0]).drop_duplicates()
        names_list = names_list.values.tolist()
        self.names_list = names_list
            
        #reading txt file: radiance
        radiance = []
        with io.open(radiance_txt_files[0], mode="r") as f:    
                next(f) #label
                next(f) #units
                next(f) #file name
                #copying data
                for line in f:
                    radiance.append(line.split())           
        radiance_files = np.array(radiance, dtype=np.float32)
        self.radiance_files = radiance_files
            
        #reading txt file: corresponding temperature data  
        temperature = []
        with io.open(temperature_txt_files[0], mode="r") as f:    
                next(f) #label
                next(f) #units
                next(f) #file name
                #copying data
                for line in f:
                    temperature.append(line.split())        
        temperature_files = np.array(temperature, dtype=np.float32)
        self.temperature_files = temperature_files
        
        #reading txt file: corresponding emissivity data  
        emissivity = []
        with io.open(emissivity_txt_files[0], mode="r") as f:    
                next(f) #label
                next(f) #units
                next(f) #file name
                #copying data
                for line in f:
                    emissivity.append(line.split())
        emissivity_files = np.array(emissivity, dtype=np.float32)
        self.emissivity_files = emissivity_files
            
    def get_fire(self):
        import pandas as pd
        import numpy as np
        from sklearn import preprocessing
        import matplotlib.pyplot as plt
        
        #Collecting particle sizes for each run
        while True:
            self.binning_choice = input("Do you have particle sizes ? Yes/No : ").lower() # This will make the user input not case-senitive
            try:
                if self.binning_choice.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
                    print("Have the sizes ready and enter them one by one as requested.")
                elif self.binning_choice.lower() in ["n","no","nope"]:
                    print("Noted.")
                else:
                    raise Exception("Invalid input! Answer can only be 'Yes' or 'No'")
            except Exception as e:
                print(e)    
            else:
                break
        
        if self.binning_choice.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
            particle_sizes = []
            for entry in self.names_list:
                while True:
                    try:
                        self.size = float(input("Size of particle in run " + str(entry) +" :")) # This will make the user input not case-senitive
                        particle_sizes.append(self.size)
                        break
                    except ValueError as e: 
                        print(e)               
       
        #Iteration through each of the runs
        Rdata = []
        Tdata = []
        Edata = []
        for column in range(0,np.shape(self.radiance_files)[1],2):         
            #Radiance 
            t_1 = pd.Series([self.radiance_files[row,column] for row in range(len(self.radiance_files[:,column]))])
            y_1 = pd.Series([self.radiance_files[row,column+1] for row in range(0,len(t_1),1)])
            dfR = pd.DataFrame({'time' : t_1, 'radiance' : y_1})
            dfR = dfR.drop_duplicates(subset=['time'], keep='last') # removing redundant time entries necessary for fitting
            dfR = dfR.drop(dfR.index[dfR['time'] < 0])              # removing negative time entries
            dfR = dfR.dropna(subset=['radiance'])
            
            #iteration adjustment to account for column differences between R and T, E datasets
            new_column=int(column*(3/2)) #temp and emissivity have 3 columns per run instead of 2 per run 
            
            #Temperature
            t_2 = pd.Series([self.temperature_files[row,new_column] for row in range(len(self.temperature_files[:,new_column]))])
            y_2 = pd.Series([self.temperature_files[row,new_column+1] for row in range(len(t_2))]) 
            dy_2 = pd.Series([self.temperature_files[row,new_column+2] for row in range(len(t_2))]) 
            dfT = pd.DataFrame({'time' : t_2, 'temperature' : y_2, 'error': dy_2})
            dfT = dfT.drop_duplicates(subset=['time'], keep='last') # removing redundant time entries necessary for fitting
            if self.error_confidence == 'y' :    
                dfT = dfT.drop(dfT.index[dfT['error'] > 1000])      # removing entries with low confidence intensity fits
            dfT = dfT.drop(dfT.index[dfT['time'] < 0])              # removing negative time entries
            dfT = dfT.drop(dfT.index[dfT['temperature'] > 9000])    # removing initial poor data points
            dfT = dfT.dropna(subset=['temperature','error'])
            
            #Emissivity
            t_3 = pd.Series([self.emissivity_files[row,new_column] for row in range(len(self.emissivity_files[:,new_column]))]) 
            y_3 = pd.Series([self.emissivity_files[row,new_column+1] for row in range(len(t_3))])
            dy_3 = pd.Series([self.emissivity_files[row,new_column+2] for row in range(len(t_3))]) 
            dfE = pd.DataFrame({'time' : t_3, 'emissivity' : y_3, 'error': dy_3})
            dfE = dfE.drop_duplicates(subset=['time'], keep='last') # removing redundant time entries necessary for fitting
            #dfE = dfE.drop(dfE.index[dfE['error'] > 0.5]) #removing emissivity inferred from poor fits
            dfE = dfE.drop(dfE.index[dfE['time'] < 0])# removing negative time entries
            dfE = dfE.dropna(subset=['emissivity','error'])
            
            #Smoothing emissivity
            import statsmodels.api as sm
            P = []  #parameter
            E = []  #error in y
            E2 = [] #error in dy/dx
            for p in np.arange(0,100):
                E_filtered = sm.nonparametric.lowess(dfE["emissivity"].to_numpy(),dfE['time'].to_numpy(), frac = p/100)
                dEdt = np.gradient(dfE["emissivity"].to_numpy(),np.log10(dfE['time'].to_numpy()))
                dEdt2 = np.gradient(E_filtered[:,1],np.log10(dfE['time'].to_numpy()))
                error = dfE['emissivity'].to_numpy() - E_filtered[:,1]
                error2 = error**2
                P.append(p/100)
                E.append(error2.mean()) 
                E2.append((dEdt.mean()- dEdt2.mean())**2)
            a = np.round(np.transpose(preprocessing.normalize([E])),3)
            b = np.round(np.transpose(preprocessing.normalize([E2])),3)
            step = []
            for i in range(0,len(a),1):
                if a[i] == b[i]:
                    step.append(0)
                else:
                    step.append(0.5)
            E_smoothed = sm.nonparametric.lowess(dfE["emissivity"].to_numpy(),dfE['time'].to_numpy(), frac = P[step.index(0.5)])
            #To account for misfitting
            if all(x >= -0.001 for x in E_smoothed[:,1]):
                E_smoothed = np.vstack(([1,0], E_smoothed))
            else:
                E_smoothed = sm.nonparametric.lowess(dfE["emissivity"].to_numpy(),dfE['time'].to_numpy(), frac = 0)
                E_smoothed = np.vstack(([1,0], E_smoothed))
          
            
            #Smoothing temperature
            P2 = []  #parameter
            T = []  #error in y
            T2 = [] #error in dy/dx
            if self.iterative_plot_choice == 'y':
                fig, ax1 =plt.subplots()
                ax1.set_xlim(5,5000)
                ax1.set_xlabel('time,ns')
                ax1.title.set_text("Fitting temperature data")
                plt.xscale('log')
                ax1.errorbar(dfT['time'].to_numpy(),dfT['temperature'].to_numpy(), yerr = dfT['error'].to_numpy(),ls='none', marker = 'o', markersize = 5, color = 'red', label = 'measured')
                #ax1.scatter(dfT['time'].to_numpy(),dfT['temperature'].to_numpy(),s=5, c='red', label = 'measured')
            for p in np.arange(0,100,1):
                T_filtered = sm.nonparametric.lowess(dfT["temperature"].to_numpy(),dfT['time'].to_numpy(), frac = p/100)
                dTdt = np.gradient(dfT["temperature"].to_numpy(),np.log10(dfT['time'].to_numpy()))
                dTdt2 = np.gradient(T_filtered[:,1],np.log10(dfT['time'].to_numpy()))
                error = dfT['temperature'].to_numpy() - T_filtered[:,1]
                error2 = error**2
                P2.append(p/100)
                T.append(error2.mean()) 
                T2.append((dTdt.mean()- dTdt2.mean())**2)       
                if self.iterative_plot_choice == 'y':
                    if p%5 == 0 and p<=30: # plotting one in twenty tries
                        ax1.plot(dfT['time'].to_numpy(), T_filtered[:,1], lw=2, label=str(p/100))
                        #ax2 = ax1.twinx()
                        #ax2.plot(dfT['time'].to_numpy(), dTdt, lw=2)
                        #ax2.plot(dfT['time'].to_numpy(), dTdt2, lw=1, label=str(p/100))
            if self.iterative_plot_choice == 'y':
                plt.gcf().set_dpi(500)
                ax1.legend()
                #ax2.legend()
                plt.show()           
            a2 = np.round(T,3)
            b2 = np.round(T2,3)
            step2 = []
            for i in range(0,len(a2),1):
                if a2[i] == b2[i]:
                    step2.append(0)
                else:
                    step2.append(0.5)
            T_smoothed = sm.nonparametric.lowess(dfT["temperature"].to_numpy(),dfT['time'].to_numpy(), frac = P2[step2.index(0.5)])
            T_smoothed = np.vstack(([1,1500], T_smoothed))
        
        
            #Interpolating radiance and smoothened emissivity
            from scipy.interpolate import PchipInterpolator as pcip
            import math
            f = pcip(dfR["time"].to_numpy(), dfR["radiance"].to_numpy(), axis=0, extrapolate=None)
            g = pcip(E_smoothed[:,0], E_smoothed[:,1], axis=0, extrapolate=None)
            h = pcip(T_smoothed[:,0], T_smoothed[:,1], axis=0, extrapolate=None)
            x_new = np.logspace(math.log10(1), math.log10(min(dfR['time'].to_numpy()[-1],dfE['time'].to_numpy()[-1],dfT['time'].to_numpy()[-1])), num=1000, endpoint=True,base=10.0)
            Ry_new = np.array(f(x_new))
            Ey_new = np.array(g(x_new))
            Ty_new = np.array(h(x_new))
            Ry_norm = np.transpose(preprocessing.normalize([Ry_new]))
            Ry_norm = Ry_norm*(Ey_new.max()/Ry_norm.max()) #Plotting ease
            
            #Plotting dataset
            if self.run_plot_choice == 'y':
                fig, ax1 =plt.subplots()
                ax1.set_xlabel('time,ns')
                ax1.title.set_text("Normalized dataset of file: " + str(self.names_list[int(column/2)][0]))
                plt.xscale('log')
                ax1.set_xlim([1,1000000])
                ax1.set_ylabel('normalized dependant variable, arb', color = 'black') 
                ax1.plot(x_new, Ry_norm, color = 'blue', label = 'radiance' )
                plt.scatter(dfE['time'].to_numpy(), dfE['emissivity'].to_numpy(), s=10, marker = 'o', color = 'green', label = 'emissivity')
                ax1.plot(x_new, Ey_new, color = 'green', label = 'emissivity fit' )
                ax1.plot(dfR['time'].to_numpy(), np.zeros_like(dfR['time'].to_numpy()), linestyle = 'dotted', color = 'black' )
            
                ax2 = ax1.twinx()
                ax2.set_ylabel('temperature, K', color = 'red')
                ax2.yaxis.label.set_color('red')
                ax2.tick_params(axis='y', colors='red')
                ax2.errorbar(dfT['time'].to_numpy(),dfT['temperature'].to_numpy(), yerr = dfT['error'].to_numpy(), ls='none', marker = 'o', markersize= 5, color = 'red', label = 'temperature data')
                ax2.plot(x_new, Ty_new, color = 'red', label = 'temperature fit' )
                ax2.plot(dfR['time'].to_numpy(), np.ones_like(dfR['time'].to_numpy())*4000, linestyle = 'dotted', color = 'red' )
                ax2.plot(dfR['time'].to_numpy(), np.ones_like(dfR['time'].to_numpy())*3000, linestyle = 'dotted', color = 'red' )
                ax2.plot(dfR['time'].to_numpy(), np.ones_like(dfR['time'].to_numpy())*2000, linestyle = 'dotted', color = 'red' )
                
                ax1.legend()
                ax1.grid(True, which='minor')
                ax1.spines['left'].set_edgecolor('blue')
                ax2.spines['right'].set_edgecolor('red')
                plt.gcf().set_dpi(500)
                plt.show()
            
            #Organizing fit curves into dataframes
            Rdata.append(Ry_new.tolist())
            Tdata.append(Ty_new.tolist())
            Edata.append(Ey_new.tolist())
            print("You have processed run "+ str(self.names_list[int(column/2)][0]) + " ~ " + str(round((((column/2)+1)/len(self.names_list))*100,1)) +" % completed")
        
        
        #binning radiance, temperature and emissivity based on particle size
        print("Binning radiance, temperature and emissivity data based on particle size")
        size_bins = [round(1.15**x) for x in np.arange(25,51)]
        rep_size = [round((size_bins[i]+size_bins[i+1])/2) for i in range(0,len(size_bins)-1,1)]
        
        Rdata = pd.DataFrame(np.array(Rdata),columns = np.array(x_new))
        Tdata = pd.DataFrame(np.array(Tdata),columns = np.array(x_new))
        Edata = pd.DataFrame(np.array(Edata),columns = np.array(x_new))
        Rdata['size'] = np.array(particle_sizes)
        Edata['size'] = np.array(particle_sizes)
        Tdata['size'] = np.array(particle_sizes)
        
        R_binned = pd.DataFrame (index = x_new)
        T_binned = pd.DataFrame (index = x_new)
        E_binned = pd.DataFrame (index = x_new)
        for bin_no in range(0,len(size_bins)-1,1):    
            Rfiltered_df = Rdata[(Rdata['size'] < size_bins[bin_no+1]) & (Rdata['size'] >= size_bins[bin_no])]
            Rfiltered_df = Rfiltered_df.drop(['size'], axis=1) 
            
            Tfiltered_df = Tdata[(Tdata['size'] < size_bins[bin_no+1]) & (Tdata['size'] >= size_bins[bin_no])]
            Tfiltered_df = Tfiltered_df.drop(['size'], axis=1)
            
            Efiltered_df = Edata[(Edata['size'] < size_bins[bin_no+1]) & (Edata['size'] >= size_bins[bin_no])]
            Efiltered_df = Efiltered_df.drop(['size'], axis=1)
            
            if Rfiltered_df.shape[0] == 0:
               R_mean = np.zeros(Rfiltered_df.shape[1])
               R_stdev = np.zeros(Rfiltered_df.shape[1])
               R_max = np.zeros(Rfiltered_df.shape[1]) 
               R_min = np.zeros(Rfiltered_df.shape[1]) 
            else:
                R_mean = Rfiltered_df.mean()
                R_mean = R_mean.to_numpy(dtype='float')
                R_stdev = Rfiltered_df.std()
                R_stdev = R_stdev.to_numpy(dtype='float')
                R_max = Rfiltered_df.max()
                R_max = R_max.to_numpy(dtype='float')
                R_min = Rfiltered_df.min()
                R_min = R_min.to_numpy(dtype='float')                
                
            if Tfiltered_df.shape[0] == 0:
               T_mean = np.zeros(Tfiltered_df.shape[1])
               T_stdev = np.zeros(Tfiltered_df.shape[1])
               T_max = np.zeros(Tfiltered_df.shape[1]) 
               T_min = np.zeros(Tfiltered_df.shape[1]) 
            else:
                T_mean = Tfiltered_df.mean()
                T_mean = T_mean.to_numpy(dtype='float')
                T_stdev = Tfiltered_df.std()
                T_stdev = T_stdev.to_numpy(dtype='float')            
                T_max = Tfiltered_df.max()
                T_max = T_max.to_numpy(dtype='float')  
                T_min = Tfiltered_df.min()
                T_min = T_min.to_numpy(dtype='float') 
                
            if Efiltered_df.shape[0] == 0:
               E_mean = np.zeros(Efiltered_df.shape[1])
               E_stdev = np.zeros(Efiltered_df.shape[1]) 
               E_max = np.zeros(Efiltered_df.shape[1])
               E_min = np.zeros(Efiltered_df.shape[1])                
            else:
                E_mean = Efiltered_df.mean()
                E_mean = E_mean.to_numpy(dtype='float')
                E_stdev = Efiltered_df.std()
                E_stdev = E_stdev.to_numpy(dtype='float')
                E_max = Efiltered_df.max()
                E_max = E_max.to_numpy(dtype='float')
                E_min = Efiltered_df.min()
                E_min = E_min.to_numpy(dtype='float')                 
                
            R_binned[str(round(rep_size[bin_no]))] = R_mean
            R_binned['error '+ str(round(rep_size[bin_no]))] = R_stdev
            R_binned['max '+ str(round(rep_size[bin_no]))] = R_max
            R_binned['min '+ str(round(rep_size[bin_no]))] = R_min

            T_binned[str(round(rep_size[bin_no]))] = T_mean
            T_binned['error '+ str(round(rep_size[bin_no]))] = T_stdev
            T_binned['max '+ str(round(rep_size[bin_no]))] = T_max
            T_binned['min '+ str(round(rep_size[bin_no]))] = T_min

            E_binned[str(round(rep_size[bin_no]))] = E_mean
            E_binned['error '+ str(round(rep_size[bin_no]))] = E_stdev
            E_binned['max '+ str(round(rep_size[bin_no]))] = E_max
            E_binned['min '+ str(round(rep_size[bin_no]))] = E_min
        
        R_binned = R_binned.replace(np.nan, 0)
        T_binned = T_binned.replace(np.nan, 0)
        E_binned = E_binned.replace(np.nan, 0)
        
        R_binned.to_csv(str(self.sample_name)+'-radiance-size-binned.txt', sep='\t')
        T_binned.to_csv(str(self.sample_name)+'-temperature-size-binned.txt', sep='\t')
        E_binned.to_csv(str(self.sample_name)+'-emissivity-size-binned.txt', sep='\t')
        
        if self.binned_plots_choice == 'y':
            ax = R_binned.reset_index().plot(x = 'index', y = [str(x) for x in rep_size], logx= True, legend = False)
            plt.title('Binned radiance')
            ax.set_xlabel('time,ns')
            ax.set_ylabel('radiance, W/Sr.m^2')
            ax.set_ylim(0,round(max(R_binned.select_dtypes(include=[np.number]).max().to_list())+0.1*max(R_binned.select_dtypes(include=[np.number]).max().to_list())))
            plt.gcf().set_dpi(500)
            
            ax = T_binned.reset_index().plot(x = 'index', y = [str(x) for x in rep_size], logx= True,legend = False)
            plt.title('Binned temperature')
            ax.set_xlabel('time,ns')
            ax.set_ylabel('temperature, K')
            ax.set_ylim(0,round(max(T_binned.select_dtypes(include=[np.number]).max().to_list())+0.1*max(T_binned.select_dtypes(include=[np.number]).max().to_list())))
            plt.gcf().set_dpi(500)
                  
            #ax = E_binned.reset_index().plot(x = 'index', y = [str(x) or x in rep_size if x>164], logx= True, legend = False, label = y)
            plt.title('Binned emissivity')
            ax.set_xlabel('time,ns')
            ax.set_ylabel('emissivity')
            ax.set_ylim(0,1)
            ax.legend()
            plt.gcf().set_dpi(500)

        return R_binned, T_binned, E_binned

    
p = Prometheus()
R_binned, T_binned, E_binned = p.get_fire()