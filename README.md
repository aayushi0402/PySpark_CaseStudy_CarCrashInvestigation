# Data Analysis on Vehicle Accidents in the US using Apache Spark
Analysis of US Vehicle Accidents. The analysis .ipynb file contains the code that was developed initially before packaging it into a Spark Application.

## Pre-requisites
1. Verifying Java Installation
2. Verifying Python installation
3. Downloading and Installing Apache Spark

## Problem Statement
Spark Application should perform below analysis and store the results for each analysis.
1. Analytics 1: Find the number of crashes (accidents) in which number of males killed are greater than 2?
2. Analysis 2: How many two wheelers are booked for crashes?
3. Analysis 3: Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy.
4. Analysis 4: Determine number of Vehicles with driver having valid licences involved in hit and run?
5. Analysis 5: Which state has highest number of accidents in which females are not involved?
6. Analysis 6: Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death
7. Analysis 7: For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
8. Analysis 8: Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)
9. Analysis 9: Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
10. Analysis 10: Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data) 

## Running the project
First you will have to fork/clone this repository. The input file is currently given in the project's input/ folder, this can be changed in the config.json depending on the input file location.
```sh
cd Spark__USCrashInvestigation
spark-submit --master "local[*]" --py-files utils --files config.json main.py
```
Running the above command
![image](https://user-images.githubusercontent.com/34810569/207420072-e6a3b915-30e2-4ffe-856e-12be764e2c3b.png)


## Sample Output
![image](https://user-images.githubusercontent.com/34810569/207656212-fc97e596-8839-40e8-b69c-dee52edff7e4.png)

Along with the ouput displayed on the terminal, the output results are stored in the locations specified in the config.json as well. You can find the sample output folders and structure in the repository and the sample output of terminal in the above screenshot.



