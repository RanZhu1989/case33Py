# case33Py 
**Distribution network plan based on the N-1 safety condition in Moded IEEE-33BW case**
>2020.11.29 v0.1   
Author：Ran Zhu @ School of Cyber Engineering and Science, SEU  
For example test  

# Example：  
![image](https://github.com/RanZhu1989/case33Py/blob/master/img/Example.png)  

# References：
- IEEE33BW case ——                 *1989, M.E.Baran, TPS*
- DER Installation Position ——    *2019, Xu, TSG*
- Load Data ——                    *Commission for Energy Regulation (CER). (2012). CER Smart Metering Project - Electricity Customer Behaviour Trial, 2009-2010 [dataset]. 1st Edition. Irish Social Science Data Archive. SN: 0012-00. www.ucd.ie/issda/CER-electricity*
- PV\WT Weather Data ——           *https://www.renewables.ninja/*
- Generation Method ——            *2016, Stefan, Energy  https://doi.org/10.1016/j.energy.2016.08.060.*
- Cable ——                        *2011, Mancarella, IET-GT*
- Brach Flow Model ——             *1989, M.E.Baran, TPS*
- Brach Flow Model-AR ——          *2013, M.Farivar, TPS*
- Directed Multi-Commodity Flow——   *2020, S.Lei, TSG*

# Data Set 
1. The load data of 1500 households in lieerick（52.6653 -8.6238) and Ennis (52.8463,-8.9807) from the Irish Power Supply Authority (half an hour granularity) are used for the load.
   ![image](https://github.com/RanZhu1989/case33Py/blob/master/img/map.png)
   ![image](https://github.com/RanZhu1989/case33Py/blob/master/img/load.png)
   
2. Weather parameters refer to the MERRA open source meteorological satellite database
3. The photovoltaic modeling is based on GSEE
4. The wind turbine modeling is based on the Vestas V90 2000 standard

# moded IEEE-33BW case  
![image](https://github.com/RanZhu1989/case33Py/blob/master/img/moded_IEEE-33BW.png)

