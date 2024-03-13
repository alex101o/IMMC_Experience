# Libraries
import numpy as np
import pandas as pd
import random
import time

# Code Runtime Check
start_time = time.time()

"""
Note: All mentions of MEI in code refers to MDI household inputs
"""

# Functions
def K_Function(Var_max, Var_min):
    """
    Summary of Function:
    Returns the K value used in logistic equation
    
    Parameters:
        Var_max (double): Relative maximum of where upper platue of the variable starts
        Var_min (double): Relative minimum of where lower platue of the variable starts
    
    Returns:
        double: K value used in logistic equation based on relative min and max values
    """
    
    # Calculating K Vallue
    return -(np.log(100/(99.9)-1)/(Var_max - Var_min))


def logistic_function(Var_max, Var_min, x):
    """
    Summary of Function:
    Returns the logistically normalized value of x based on relative min and max values.
    Upper asymptote at y = 100, lower asymptote at y = 0
    
    Parameters:
        Var_max (double): Relative x location of where upper platue of the variable starts
        Var_min (double): Relative x location of where lower platue of the variable starts
        x (double): Input value that is normalized
    
    Returns:
        double: Logistically normalized value of x based on specific upper and lower set platue regions
    """
    # Calculating Logistic Normalization of x
    return (100/(1+np.exp(-K_Function(Var_max, Var_min)*(x-Var_min))))

def Inverse_Logistic_Function(Var_max, Var_min, x):
    """
    Summary of Function:
    Returns the inverse logistically normalized value of age input from household pet owner.
    Upper asymptote at y = 100, lower asymptote at y = 0

    Parameters:
        Var_max (double): Relative x location of where upper platue of the age input starts
        Var_min (double): Relative x location of where lower platue of the age input starts
        x (double): Age of official owner of pet of household
    
    Returns:
        double: Inverse logistically normalized value of x based on specific upper and lower set platue regions
    """
    
    return (200/(1+np.exp((np.log(200/(99.9+100)-1)/(Var_max - Var_min))*(-x+Var_max)))) - 100

def Squared_Function(x):
    """
    Summary of Function:
    Returns the squared normalized value of time flexibility input from househol pet owner
    
    Parameters:
        x (double): Flexibility input (0-100) of household
    
    Returns:
        double: Squared normalized value of x
    """
    
    return ((x/100)**2)*100

def Main_Section_1(animal_folder, family_file):
    """
    Summary of Function:
    Section 1 of model. Check if household meets basic requirements of pet factors before sending to 
    section 2. Sends failing/succeeding households information to main function.
    
    Parameters:
        animal_folder (str): Folder name of location with csv files of weights for pet factor layer,
            wieghts for output layer, and  upper & lower values for variables using logistical and 
            inverse logistical normalization functions.
        family_file (str): csv file name of dataframe of households inputs for model
    
    Returns:
        Fam_Bool_List (list: boolean): List of ordered boolean values of families that do and don't 
        make basic requirements of MEI, LS, AOO, and ET required from specific pet.
        count (int): Count of total families that fail to meet basic requirements of pet species
    """
    # Dataframe for upper and lower maximums and minimums of where logistic normalization platue 
    # starts for MEI, ET, LS, CH, PK, AOO 
    # CSV Example view in Appendix
    Pet_Requirements_Data = pd.read_csv(animal_folder + "/Upper Lower.csv", index_col=0)

    # Aquires household inputs dataframe
    Family_Inputs = pd.read_csv(family_file, index_col=False)
    
    # Creates boolean list for families that fail basic requirements
    Fam_Bool_List = [True]*(len(Family_Inputs.columns)-1)

    # Checks individual household for meeting certain basic pet requirements
    for i in range(len(Fam_Bool_List)):
        if Pet_Requirements_Data["MEI"][0] > Family_Inputs.iloc[0][i+1]:
            Fam_Bool_List[i] = False
        if Pet_Requirements_Data["LS"][0] > Family_Inputs.iloc[3][i+1]:
            Fam_Bool_List[i] = False
        if Pet_Requirements_Data["AOO"][0] > Family_Inputs.iloc[8][i+1]:
            Fam_Bool_List[i] = False
        if Pet_Requirements_Data["ET"][0] > Family_Inputs.iloc[1][i+1]:
            Fam_Bool_List[i] = False
    
    # Counter for number of families that fail section 1 of model
    count = 0
    for i in Fam_Bool_List:
        if i == False:
            count += 1
            
    # Return
    return Fam_Bool_List, count
    
def Main_Section_2(animal_folder, family_file):
    """
    Summary of Function:
    Section 2 of model. Check if combination of household inputs allow a compatibility score of over 
    65 after running through normalization, standardization, pet factor, and output layer. Sends 
    compatibility score of each households to main function.
    
    Parameters:
        animal_folder (str): Folder name of location with csv files of weights for pet factor layer,
            wieghts for output layer, and  upper & lower values for variables using logistical and 
            inverse logistical normalization functions
        family_file (str): csv file name of dataframe of households inputs for model
    
    Returns:
        Final_Evaluation (List: int): List of ordered compatibility scores of each households (1-100)
        count (int): Total number of househols that fail to meet compatibility score of 65 (threshold)
    """
    
    # Dataframe for upper and lower maximums and minimums of where logistic normalization platue starts 
    # for MEI, ET, LS, CH, PK, AOO
    # Example CSV in Appendix
    Cat_Requirements_Data = pd.read_csv(animal_folder + "/Upper Lower.csv", index_col=0)

    # Dataframe for Time Intensity Weights of ET, LS, MEI, PK, PC for specific pet species
    # Example CSV in Appendix
    Cat_Time_Intensity_Percentage = pd.read_csv(animal_folder + "/Time Intensity.csv")

    # Dataframe for Cost of Living Weights of MEI, LS, TF, CH for specific pet species
    # Example CSV in Appendix
    Cat_Cost_of_Living_Percentage = pd.read_csv(animal_folder + "/Cost of Living.csv")

    # Dataframe for Physical Intensity Weights of AOO, PMD for specific pet species
    # Example CSV in Appendix
    Cat_Physical_Intensity_Percentage = pd.read_csv(animal_folder + "/Physical Intensity.csv")

    # Dataframe for Pet Training Weights for ET, MEI, LS, PK for specific pet species
    # Example CSV in Appendix
    Cat_Pet_Training_Percentage = pd.read_csv(animal_folder + "/Pet Training.csv")

    # Dataframe for Owner Capatibility Weights for Cost of Living, Time Intensity, 
    # Pet Training, Physical Intensity for specific pet species
    # Example CSV in Appendix
    Owner_Capability_Percentage = pd.read_csv(animal_folder + "/Owner Capatibility.csv")

    # Aquires household inputs dataframe
    # Example CSV in Appendix
    Family_Inputs = pd.read_csv(family_file, index_col=False)

    # Creates empty household inputs to normalize and standardize (1-100) dataframe:
    #   Col: Different families (num of columns in Family_Inputs dataframe)
    #   Row: Different household input variables (MEI, Et, TF, LS, PMD, PC, CH, PK, AOO)
    index_labels = ["MEI", "ET", "TF", "LS", "PMD", "PC", "CH", "PK", "AOO"]
    family_results = pd.DataFrame(index=index_labels, columns = range(len(Family_Inputs.columns)-1))

    # Normalize and Standardize MEI variable for household input variables
    # Logistic normalization
    # Standardize (1-100)
    # Set to row 1 of family_results dataframe
    index = 1-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(logistic_function(Cat_Requirements_Data["MEI"][1], Cat_Requirements_Data["MEI"][0], Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Normalize and Standardize ET variable for household input variables
    # Logistic normalization
    # Standardize (1-100)
    # Set to row 2 of family_results dataframe
    index = 2-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(logistic_function(Cat_Requirements_Data["ET"][1], Cat_Requirements_Data["ET"][0], Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Normalize and Standardize TF variable for household input variables
    # Squared normalization
    # Standardized (1-100)
    # Set to row 3 of family_results dataframe
    index = 3-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(Squared_Function(Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Normalize and Standardize LS variable for household input variables
    # Logistic normalization
    # Standardize (1-100)
    # Set to row 4 of family_results dataframe
    index = 4-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(logistic_function(Cat_Requirements_Data["LS"][1], Cat_Requirements_Data["LS"][0], Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Normalize and Standardize PMD variable for household input variables
    # No normalization
    # Already standardized (1-100)
    # Set to row 5 of family_results dataframe
    index = 5-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(Family_Inputs.iloc[index][i+1])
    family_results.iloc[index] = things_list

    # Normalize and Standardize PC variable for household input variables
    # No normalization
    # Already standardized (1-100)
    # Set to row 6 of family_results dataframe
    index = 6-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(Family_Inputs.iloc[index][i+1])
    family_results.iloc[index] = things_list

    # Normalize and Standardize CH variable for household input variables
    # Logistic normalization
    # Standardize (1-100)
    # Set to row 7 of family_results dataframe
    index = 7-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(logistic_function(Cat_Requirements_Data["CH"][1], Cat_Requirements_Data["CH"][0], Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Normalize and Standardize PK variable for household input variables
    # Logistic normalization
    # Standardize (1-100)
    # Set to row 8 of family_results dataframe
    index = 8-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        things_list.append(logistic_function(Cat_Requirements_Data["PK"][1], Cat_Requirements_Data["PK"][0], Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Normalize and Standardize AOO variable for household input variables
    # Inverse logistic normalization
    # Standardize (1-100)
    # Set to row 9 of family_results dataframe
    index = 9-1
    things_list = []
    for i in range(len(Family_Inputs.iloc[index])-1):
        if Family_Inputs.iloc[index][i+1] > 70:
            things_list.append(0)
        else:
            things_list.append(Inverse_Logistic_Function(Cat_Requirements_Data["AOO"][1], Cat_Requirements_Data["AOO"][0], Family_Inputs.iloc[index][i+1]))
    family_results.iloc[index] = things_list

    # Empty pet factors standardized (1-100) score dataframe:
    #   Col: Different families (num of columns in Family_Inputs dataframe)
    #   Row: Different pet factors (Time Intensity, Cost of Living, Physical Intensity, Pet Training)
    Family_Pet_Factors = pd.DataFrame(index=["Time Intensity", "Cost of Living", "Physical Intensity", "Pet Training"], columns = range(len(Family_Inputs.columns)-1))

    # Standardized time intensity score of each household using weights for ET, LS, MEI, PK, PC
    # Already standardized (1-100)
    # Set to row 1 of Family_Pet_Factors dataframe
    Time_Intensity = []
    for i in range(len(family_results.iloc[0])):
        sum = 0
        for ii in Cat_Time_Intensity_Percentage.columns.values.tolist():
            sum += (family_results.iloc[:, i].loc[ii]*Cat_Time_Intensity_Percentage[ii][0])
        Time_Intensity.append(sum)
    Family_Pet_Factors.iloc[0] = Time_Intensity

    # Standardized cost of living score of each household using weights for MEI, LS, TF, CH
    # Already standardized (1-100)
    # Set to row 1 of Family_Pet_Factors dataframe
    Cost_of_Living = []
    for i in range(len(family_results.iloc[0])):
        sum = 0
        for ii in Cat_Cost_of_Living_Percentage.columns.values.tolist():
            sum += (family_results.iloc[:, i].loc[ii]*Cat_Cost_of_Living_Percentage[ii][0])
        Cost_of_Living.append(sum)
    Family_Pet_Factors.iloc[1] = Cost_of_Living

    # Standardized physical intensity score of each household using weights for PMD, AOO
    # Already standardized (1-100)
    # Set to row 1 of Family_Pet_Factors dataframe
    Physical_Intensity = []
    for i in range(len(family_results.iloc[0])):
        sum = 0
        for ii in Cat_Physical_Intensity_Percentage.columns.values.tolist():
            sum += (family_results.iloc[:, i].loc[ii]*Cat_Physical_Intensity_Percentage[ii][0])
        Physical_Intensity.append(sum)
    Family_Pet_Factors.iloc[2] = Physical_Intensity

    # Standardized pet training score of each household using weights for ET, MEI, LS, PK
    # Already standardized (1-100)
    # Set to row 1 of Family_Pet_Factors dataframe
    Pet_Training = []
    for i in range(len(family_results.iloc[0])):
        sum = 0
        for ii in Cat_Pet_Training_Percentage.columns.values.tolist():
            sum += (family_results.iloc[:, i].loc[ii]*Cat_Pet_Training_Percentage[ii][0])
        Pet_Training.append(sum)
    Family_Pet_Factors.iloc[3] = Pet_Training

    # List of ordered final evaluation score of each household using weights for
    # Time Intensity, Cost of Living, Physical Intensity, Pet Training in
    # Final_Evaluation
    Final_Evaluation = []
    for i in range(len(Family_Pet_Factors.iloc[0])):
        sum = 0;
        for ii in Owner_Capability_Percentage.columns.values.tolist():
            sum += (Family_Pet_Factors.iloc[:, i].loc[ii]*Owner_Capability_Percentage[ii][0])
        Final_Evaluation.append(sum)
        
    # Counter for number of families that fail to meet 65 threshhold compatibility score
    count = 0
    for i in Final_Evaluation:
        if i < 65:
            count += 1
    
    # Return
    return Final_Evaluation, count


def Main(animal_folder, family_file, name, death_list = [], has_pet = [], retention = [], age = 0):
    """
    Summary of Function:
    Saves to csv file of capatibility numbers (1-100), average capatibility score, fail 
    counters for section 1, section 2 and final evaluation, death counter, proportion 
    of households who have retained their pet through all years, and average 
    years of retention of every family for a specific species of pet
    
    Parameters:
        animal_folder (str): Folder name of location with csv files of weights for pet factor layer,
            wieghts for output layer, and  upper & lower values for variables using logistical and 
            inverse logistical normalization functions
        family_file (str): csv file name of dataframe of households inputs for model
        name (str): csv file name to save final dataframe to
        death_list (List: boolean): Ordered boolean list of households with official owner age over 80 
        (persumebly dead/too old to take care of pet). Length = total number of households in 
        family_file csv
        haspet (List: boolean): Previous year's ordered boolean list of households who has the pet. 
        Length = total number of households in family_file csv
        retention (List: int, str): Previous year's ordered integer list of households of years having 
        the pet (String: N/A for families that never had pet to begin with). Length = total number 
        of households in family_file csv
        age (int): Change in year of retention from individuals who's kept the pet
    
    Returns:
        has_pet (List: boolean): This year's ordered boolean list of households who has the pet Length 
        = total number of households in family_file csv
        retention (List: int, str): This year's ordered integer list of households of years having the 
        pet (String: N/A for families that never had pet to begin with). Length = total number of 
        households in family_file csv
        prop_has_pet (double): Proportion of households that currently have the pet species out of all
        households that could've at some point have that pet
        Average_Retention (double): This's years average number of years each household who could've
        had the pet at some point had it for
        Average_Compatibility_Score (double): Average compatibility score of all families that have
        a compatibility score
    """
    # Variables
    set_retention = False
    Final_Evaluation_Count = 0
    death_count = 0
    retention_sum = 0
    retention_count = 0
    has_pet_count = 0
    compatibility_sum = 0
    compatibility_count = 0
    
    # Run household inputs through section 1 and 2 of model
    Fam_Bool_List, sec_1_fail_count = Main_Section_1(animal_folder, family_file)
    Final_Evaluation_List, sec_2_fail_count = Main_Section_2(animal_folder, family_file)
    
    # Saves evaluation score to Final_Evaluation_Data dataframe
    Final_Evaluation_Data = pd.DataFrame()
    Final_Evaluation_Data["Index"] = ["Section 1 Evaluation", "Section 2: Compatibility Score Evaluation", "FINAL EVALUATION", "Deaths", "Has Pet", "Retention", 
                                      "Average Compatibility Score"]
    
    # Creates empty list of death list of correct length if none is provided
    if death_list == []:
        death_list = [False] * len(Final_Evaluation_List)
    
    # Checks if retention list is properly setup
    if retention == []:
        set_retention = True
            
    # Starts eveluation of every household for pet compatibility and gathering of other data
    for i in range(len(Final_Evaluation_List)):  
        
        # Sets or change has_pet list to correct bool value for each family based on compatibility score
        # Sets or updates retention list to correct int value for each family based on how long they've
        # stayed above threshold compatibility score
        if not(Fam_Bool_List[i] and Final_Evaluation_List[i]>65 and not death_list[i]):
            if set_retention:
                retention.append("N/A")
                has_pet.append(False)
            else:
                has_pet[i] = False
        else:
            if set_retention:
                retention.append(0)
                has_pet.append(True)
            else:
                if retention[i] != "N/A":
                    if has_pet[i]:
                        retention[i] += age
        
        # Counter for households who fail overall evaluation
        if not(Fam_Bool_List[i] and Final_Evaluation_List[i]>65 and not death_list[i]):
            Final_Evaluation_Count += 1
        
        # Counter for households who's official owner age is over 80 (persumebly dead/unable to take care of pet)
        if death_list[i]:
            death_count += 1
    
    
    # Counter for poeple who's about to obtain pet
    # Summation of total retention of all households who owned the pet species at some point
    for i in retention:
        if i != "N/A":
            retention_count += 1
            retention_sum += i
    
    # Calculates average retention of every household who owned the pet species at some point
    # If no household obtained the pet species, Average_Retention set to "No one had this pet
    # spiecies"
    if retention_count == 0:
        Average_Retention = "No one had this pet species"
    else:
        Average_Retention = retention_sum/retention_count
    
    # Counter for households that have the pet species currently 
    for i in has_pet:
        if i:
            has_pet_count += 1

    # Calculates proportion of households that currently has pet species out of all households
    # who previously had it
    # If no one had pet species, prop_has_pet set to "No one had this pet species"
    if retention_count == 0:
        prop_has_pet = "No one had this pet species"
    else:
        prop_has_pet = has_pet_count/retention_count
    
    # Counts total number of households with compatibility score
    # Summation of total compatibility score of all households
    for i in Final_Evaluation_List:
        compatibility_count += 1
        compatibility_sum += i
        
    # Finds average compatibility score of all households
    Average_Compatibility_Score = compatibility_sum/compatibility_count
    
    # Adds all evalutation information, has_pet, and retention about household to Final_Evaluation_Data dataframe
    for i in range(len(Final_Evaluation_List)):  
        Final_Evaluation_Data["Household " + str(i)] = [Fam_Bool_List[i], Final_Evaluation_List[i],  (Fam_Bool_List[i] and Final_Evaluation_List[i]>65 and not 
                                                                                                      death_list[i]), death_list[i], has_pet[i], retention[i], 0]
    
    # Saves all extra data in Final_Evaluation_Data dataframe
    Final_Evaluation_Data.insert(1, "Fail Counters, etc.", [sec_1_fail_count, sec_2_fail_count, Final_Evaluation_Count, death_count, prop_has_pet, Average_Retention, Average_Compatibility_Score])

    # Saves Final_Evaluation_Data dataframe to csv in speicifc animal folder
    Final_Evaluation_Data.to_csv(animal_folder + "/" + animal_folder + " " + name + " Results.csv", index=False)
    return has_pet, retention, prop_has_pet, Average_Retention, Average_Compatibility_Score, Final_Evaluation_Count

# Run 6 Test Families in model for Task 1 and Task 2 for 5 pet species
Main("Cat", "Sample families 1.csv", "Test Pet Sample Families")
Main("Dog", "Sample families 1.csv", "Test Pet Sample Families")
Main("Horse", "Sample families 1.csv", "Test Pet Sample Families")
Main("Snake", "Sample families 1.csv", "Test Pet Sample Families")
Main("Turtle", "Sample families 1.csv", "Test Pet Sample Families")


# Population Simulation

# Income and Discretionary Income Generator
def MEI_Distribution(Num_Trials, country, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated income and discretionary income for that year and two time 
    dimentions afterwards.
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aging (List: int): Age benchmarks we age individuals without having the pet at to save income data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save income data to csv
    
    Returns:
    year_mei_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by discretionary income of X households generated
    year_income_List (3D List: double): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by income of X households generated
    """
    
    # Constants
    GROSS_SAVINGS_RATE_US = .178
    GROSS_SAVINGS_RATE_BRAZIL = .174
    GROSS_SAVINGS_RATE_US_ETHIOPIA = .2156
    US_INCOME_INCREASE = 0.015
    BRAZIL_INCOME_INCREASE = 0.03
    ETHIOPIA_INCOME_INCREASE = 0.027
    SMALL_CRASH_PERCENTAGE = 0.15
    BIG_CRASH_PERCENTAGE = 0.05
    
    # Lists
    year_mei_list = []
    age_mei_List = []
    mei_List = []
    year_income_List = []
    age_income_list = []
    income_list = []
    
    # Generates 1D array for total number of households for income 
    # and discretionary income
    df = pd.read_csv(country + "/Income Distribution.csv", index_col=False)
    for i in range(Num_Trials):
        rand = random.uniform(0, 100)
        for ii in range(len(df.index)):
            if rand < df["Percent"][ii]:
                income = random.uniform(df["Low"][ii], df["High"][ii])
                income_list.append(income)
                if(country == "US"):
                    mei_List.append(GROSS_SAVINGS_RATE_US * income)
                if(country == "Brazil"):
                    mei_List.append(GROSS_SAVINGS_RATE_BRAZIL * income)
                if(country == "Ethiopia"):
                    mei_List.append(GROSS_SAVINGS_RATE_US_ETHIOPIA * income)
                break
    
    # Appends
    age_income_list.append(np.array(income_list))
    year_income_List.append(age_income_list)
    age_mei_List.append(mei_List)
    year_mei_list.append(age_mei_List)
    
    # Generate 2D array of every year we age them to by total number of households
    # for income and discretionary income
    if aging != [0]:
        for ii in range(max(aging)):
            age_income_list = []
            for i in range(Num_Trials):
                crash = random.uniform(0, 1)
                if crash > SMALL_CRASH_PERCENTAGE:
                    if country == "US":
                        raises = np.random.normal(US_INCOME_INCREASE, 0.005, 1)[0]
                        if raises < 0:
                            raises = 5
                    if country == "Brazil":
                        raises = np.random.normal(BRAZIL_INCOME_INCREASE, 0.005, 1)[0]
                        if raises < 0:
                            raises = 0
                    if country == "Ethiopia":
                        raises = np.random.normal(ETHIOPIA_INCOME_INCREASE, 0.005, 1)[0]  
                        if raises < 0:
                            raises = 0
                    income_list[i] = income_list[i] * (1 + raises)
                elif crash > BIG_CRASH_PERCENTAGE:
                    income_list[i] = income_list[i] * (0.9)
                else:
                    income_list[i] = income_list[i] * (0.75)
                    
            for iii in aging:
                if ii != 0 and ii == iii-1:
                    age_income_list = np.append(age_income_list, income_list) 
                    year_income_List = year_income_List + [[age_income_list]]

                    mei_List = []
                    age_mei_List = []
                    for iiii in income_list:
                        if(country == "US"):
                            mei_List.append(GROSS_SAVINGS_RATE_US * iiii)
                        if(country == "Brazil"):
                            mei_List.append(GROSS_SAVINGS_RATE_BRAZIL * iiii)
                        if(country == "Ethiopia"):
                            mei_List.append(GROSS_SAVINGS_RATE_US_ETHIOPIA * iiii)
                    age_mei_List.append(mei_List)
                    year_mei_list.append(age_mei_List) 
    
    
    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for income and discretionary income
    if year != [0]:
        for i in range(len(year_mei_list)):
            age_income_list = year_income_List[i]
            income_list = np.append([], age_income_list[0])
            age_mei_List = year_mei_list[i]
            for ii in range(max(year)):
                for iiii in range(Num_Trials):
                    crash = random.uniform(0, 1)
                    if crash > SMALL_CRASH_PERCENTAGE:
                        if country == "US":
                            raises = np.random.normal(US_INCOME_INCREASE, 0.005, 1)[0]
                            if raises < 0:
                                raises = 0
                        if country == "Brazil":
                            raises = np.random.normal(BRAZIL_INCOME_INCREASE, 0.005, 1)[0]
                            if raises < 0:
                                raises = 0
                        if country == "Ethiopia":
                            raises = np.random.normal(ETHIOPIA_INCOME_INCREASE, 0.005, 1)[0]  
                            if raises < 0:
                                raises = 0
                        income_list[iiii] = income_list[iiii] * (1 + raises)
                    elif crash > BIG_CRASH_PERCENTAGE:
                        income_list[iiii] = income_list[iiii] * (0.9)
                    else:
                        income_list[iiii] = income_list[iiii] * (0.75)
                for iii in range(len(year)):
                    if ii != 0 and ii == year[iii]-1:
                        age_income_list = age_income_list + [income_list]
                        mei_List = []
                        for iiii in income_list:
                            if(country == "US"):
                                mei_List.append(GROSS_SAVINGS_RATE_US * iiii)
                            if(country == "Brazil"):
                                mei_List.append(GROSS_SAVINGS_RATE_BRAZIL * iiii)
                            if(country == "Ethiopia"):
                                mei_List.append(GROSS_SAVINGS_RATE_US_ETHIOPIA * iiii)
                        age_mei_List.append(mei_List)
                        year_mei_list[i] = age_mei_List
                        year_income_List[i] = age_income_list

    # Return
    return year_mei_list, year_income_List
                
                
# Income and Discretionary Income Generator
def AOO_Distribution(Num_Trials, country, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated age for that year and two time dimentions afterwards.
    Provides a ordered 3D array of death boolean to detect whether age of household is above 80 (
    persumed dead/unable to take care of pet)
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aging (List: int): Age benchmarks we age individuals without having the pet at to save age data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save age data to csv
    
    Returns:
    year_aoo_list (3D List: int) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by age of X households generated
    year_death_list (3D List: boolean): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by death bool of X households
    """
    
    # Lists
    year_aoo_list = []
    age_aoo_List = []
    aoo_List = []
    year_death_list = [[[False] * Num_Trials] * len(aging)] * len(year)
    year_death_list = np.array(year_death_list)
    
    # Generates 1D array for total number of households for age
    df = pd.read_csv(country + "/Age Distribution.csv", index_col=False)
    for i in range(Num_Trials):
        rand = random.uniform(1, 100)
        for ii in range(len(df.index)):
            if rand < df["Percentage"][ii]:
                age = random.randint(df["From"][ii], df["To"][ii])
                if(country == "US"):
                    aoo_List.append(age)
                if(country == "Brazil"):
                    aoo_List.append(age)
                if(country == "Ethiopia"):
                    aoo_List.append(age)
                break    
    
    # Appends
    age_aoo_List = [np.append(age_aoo_List, aoo_List)]
    year_aoo_list = age_aoo_List
    
    # Generate 2D array of every year we age them to by total number of households
    # for age
    if aging != [0]:
        for ii in range(len(aging)):
            if aging[ii] != 0:
                thing = year_aoo_list[0]
                age_aoo_List = []
                for i in range(Num_Trials):
                    age_aoo_List.append(thing[i] + aging[ii])
                    if(age_aoo_List[i] > 80):
                        year_death_list[0][ii][i] = True
                        age_aoo_List[i] = 80
                year_aoo_list.append(age_aoo_List)
    
    # Append
    year_aoo_list = [year_aoo_list]

    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for income and discretionary income
    if year != [0]:
        for ii in range(len(year)):
            if year[ii] != 0:
                age_aoo_List = []
                for i in range(len(year_aoo_list[0])):
                    ages = []
                    for iii in range(Num_Trials):
                        agey = year_aoo_list[0][i][iii] + year[ii]
                        if(agey > 80):
                            year_death_list[ii][i][iii] = True
                            agey = 80       
                        ages.append(agey)
                    age_aoo_List.append(ages)
                year_aoo_list.append(age_aoo_List)
    # Return
    return year_aoo_list, year_death_list

def PK_Distribution(Num_Trials, country, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated PK value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aging (List: int): Age benchmarks we age individuals without having the pet at to save PK value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save PK value data to csv
    
    Returns:
    year_pk_list (3D List: int) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by PK value of X households generated
    """
    
    # Lists
    pk_list = []
    age_pk_list = []
    year_pk_list = []
    
    # Generates 1D array for total number of households for PK value
    for i in range(Num_Trials):
        pk = random.randint(0, 100)
        pk_list.append(pk)
        
    # Generate 2D array of every year we age them to by total number of households
    # for PK value
    for i in aging:
        age_pk_list = []
        age_pk_list = [np.append(age_pk_list, pk_list)]
        year_pk_list.append(age_pk_list)
    
    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for PK value
    for i in range(len(aging)):
        for ii in range(len(year)):
            if ii == 0:
                continue
            else:
                pk_list = []
                for iii in range(Num_Trials):
                    pk = 100
                    pk_list.append(pk)
            year_pk_list[i].append(pk_list)
    
    # Return
    return year_pk_list

def ET_Distribution(Num_Trails, country, aoo, mei, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated ET value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aoo (3D List: int): Previously generated 3D age list of households
    mei (3D List: double): Previously generated 3D income of households
    aging (List: int): Age benchmarks we age individuals without having the pet at to save ET value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save ET value data to csv
    
    Returns:
    year_et_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by ET value of X households generated
    """
    
    # List
    year_et_list = []
    age_et_list = []
    
    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for ET value
    dfaoo = pd.read_csv(country + "/ETime Age Distribution.csv", index_col=False)
    dfmei = pd.read_csv(country + "/ETime Income Distribution.csv", index_col=False)
    for iiiii in range(len(year)):
        age_et_list = []
        for iiii in range(len(aging)):
            et_list= []
            for i in range(Num_Trails):
                age = aoo[iiii][iiiii][i]
                income = mei[iiii][iiiii][i]
                for ii in range(len(dfaoo.index)):
                    if age <= dfaoo["Age To"][ii]:
                        agetime = dfaoo["Time"][ii]
                        break
                for iii in range(len(dfmei.index)):
                    if income < dfmei["Money To"][iii]:
                        moneytime = dfmei["Time"][iii]
                        break
                time = (agetime + moneytime)/2 + np.random.normal(0, 0.2, 1)[0]
                et_list.append(time)
            age_et_list.append(et_list)
        year_et_list.append(age_et_list)
    
    # Return
    return year_et_list 
        

def TF_Distribution(Num_Trails, country, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated TF value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aging (List: int): Age benchmarks we age individuals without having the pet at to save TF value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save TF value data to csv
    
    Returns:
    year_tf_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by TF value of X households generated
    """
    
    # Constants
    USTFVAL = 74
    BRAZILTFVAL = (100-((100-USTFVAL)/4*5))
    ETHIOPIATFVAL = USTFVAL
    
    # Lists
    tf_list = []
    age_tf_list = []
    year_tf_list = []
    
    
    for i in range(Num_Trails):
        if country == "US":
             tf = np.random.normal(USTFVAL, 7, 1)[0]
             if tf > 100:
                 tf = random.uniform(80, 101)
        if country == "Brazil":
             tf = np.random.normal(BRAZILTFVAL, 7, 1)[0]
             if tf > 100:
                 tf = random.uniform(80, 101)
        if country == "Ethiopia":
             tf = np.random.normal(ETHIOPIATFVAL, 7, 1)[0]
             if tf > 100:
                 tf = random.uniform(80, 101)
        tf = int(tf)
        tf_list.append(tf)
        
    
    for i in aging:
        age_tf_list.append(tf_list)
    
    
    for i in year:
        year_tf_list.append(age_tf_list)
        
    # Return
    return year_tf_list
       
def LS_Distribution(Num_Trails, country, mei, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated LS value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    mei (3D List: double): Previously generated 3D income of households
    aging (List: int): Age benchmarks we age individuals without having the pet at to save LS value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save LS value data to csv
    
    Returns:
    year_ls_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by LS value of X households generated
    """
    
    # Constants
    USLSMEI = 1.801
    BRAZILLSMEI = 0.371
    ETHIOPIALSMEI = 0.571
    RENTPROP = 0.3
    
    # List
    age_ls_list = []
    year_ls_list = []
    
    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for LS value
    for iii in range(len(mei)):
        age_ls_list = []
        for ii in range(len(mei[iii])):
            ls_list = []
            for i in range(Num_Trails):
                if country == "US":
                    ls = USLSMEI * (mei[iii][ii][i]*RENTPROP)
                if country == "Brazil":
                    ls = BRAZILLSMEI * (mei[iii][ii][i]*RENTPROP)
                if country == "Ethiopia":
                    ls = ETHIOPIALSMEI * (mei[iii][ii][i]*RENTPROP)
                ls = int(ls)
                ls_list.append(ls)
            age_ls_list.append(ls_list)
        year_ls_list.append(age_ls_list)
        
    # Return
    return year_ls_list

def PMD_Distribution(Num_Trails, country, aoo, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated PMD value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aoo (3D List: int): Previously generated 3D age list of households
    aging (List: int): Age benchmarks we age individuals without having the pet at to save PMD value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save PMD value data to csv
    
    Returns:
    year_ls_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by PMD value of X households generated
    """
    
    # List
    age_pmd_list = []
    year_pmd_list = []

    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for PMD value
    for iii in range(len(aoo)):
        age_pmd_list = []
        for ii in range(len(aoo[iii])):
            pmd_list = []
            for i in range(Num_Trails):
                if aoo[iii][ii][i] < 35:
                    pmd = 100 + np.random.normal(0, 5, 1)[0]
                else:
                    pmd = y = -2.222 * aoo[iii][ii][i] + 177.77 + np.random.normal(0, 5, 1)[0]
                if pmd > 100:
                    pmd = 100
                if pmd < 0:
                    pmd = random.randint(0, 10)
                pmd = int(pmd)
                pmd_list.append(pmd)
            age_pmd_list.append(pmd_list)
        year_pmd_list.append(age_pmd_list)
    
    # Return
    return year_pmd_list

def PC_Distribution(Num_Trails, country, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated PC value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aging (List: int): Age benchmarks we age individuals without having the pet at to save PC value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save PC value data to csv
    
    Returns:
    year_pc_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by PC value of X households generated
    """
    
    # List
    pc_list = []
    age_pc_list = []
    year_pc_list = []
    
    
    for i in range(Num_Trails):
        pc = np.random.normal(50, 12.5, 1)[0]
        if pc < 0:
            pc = 0
        if pc > 100:
            pc = 100
        pc = int(pc)
        pc_list.append(pc)
    
    
    for ii in range(len(aging)):
        age_pc_list.append(pc_list)
        
    
    for ii in range(len(year)):
        year_pc_list.append(age_pc_list)
        
    # Return
    return year_pc_list
                    
def CH_Distribution(Num_Trails, country, aoo, mei, aging = [0], year = [0]):
    """
    Summary of Function:
    Provides a 3D array of randomly generated CH value for that year and two time dimentions afterwards
    (100 after one year having that pet for time dimention with that pet).
    
    Parameters:
    Num_Trials (int): Number of households we want to generate
    country (str): Country from which we want to generate households from (US, Brazil, Ethiopia)
    aoo (3D List: int): Previously generated 3D age list of households
    mei (3D List: double): Previously generated 3D income of households
    aging (List: int): Age benchmarks we age individuals without having the pet at to save CH value data to csv
    year (List: int): Year benchmarks we age individuals having the pet at to save CH value data to csv
    
    Returns:
    year_ch_list (3D List: double) (X by Y by Z): Aging (aging) dimention of household without per by aging (year)
    dimention of households with pet by CH value of X households generated
    """
    
    # List
    age_ch_list = []
    year_ch_list = []
    
    # Creates 3D array of every year we age them to by years we simulate in the future 
    # with pet by total number of households for CH value
    for iii in range(len(aoo)):
        age_ch_list = []
        for ii in range(len(aoo[iii])):
            ch_list = []
            for i in range(Num_Trails):
                ch1 = 0.11 * mei[iii][ii][i] + 190
                ch2 = 6.875* aoo[iii][ii][i] + 300
                ch = ((.8 * ch1) + (.2 * ch2))/2 + np.random.normal(0, 20, 1)[0]
                if ch > 850:
                    ch = 850
                if ch < 300:
                    ch = 300
                ch = int(ch)
                ch_list.append(ch)
            age_ch_list.append(ch_list)
        year_ch_list.append(age_ch_list)
    
    # Return
    return year_ch_list

# Generate num_fam Family Inputs for country on certain animal species
# extended to certain age in future without pet and certain age in 
# future with pet
def Create_Fam(num_fam, country, animal, aging = [0], year = [0]):
    # Generating values
    mei, income = MEI_Distribution(num_fam, country, aging, year)
    aoo, death = AOO_Distribution(num_fam, country, aging, year)
    et = ET_Distribution(num_fam, country, aoo, income, aging, year)
    tf = TF_Distribution(num_fam, country, aging, year)
    ls = LS_Distribution(num_fam, country, income, aging, year)
    pmd = PMD_Distribution(num_fam, country, aoo, aging, year)
    pc = PC_Distribution(num_fam, country, aging, year)
    ch = CH_Distribution(num_fam, country, aoo, income, aging, year)
    pk = PK_Distribution(num_fam, country, aging, year)
    
    # Lists
    aging_list = []
    year_list = []
    propretention_list = []
    avgretention_list = []
    averagescore_list =[]
    eval_fail_list =[]
    
    # Simulate
    for ii in range(len(aging)):
        list1 = []
        list2 = []
        for i in range(len(year)):
            if i == 0:
                age = year[i]
            else:
                age = year[i] - year[i-1]
            df = pd.DataFrame(index = ["MEI", "ET", "TF", "LS", "PMD", "PC", "CH", "PK", "AOO"], columns = range(num_fam))
            df.iloc[0] = mei[ii][i]
            df.iloc[1] = et[ii][i]
            df.iloc[2] = tf[ii][i]
            df.iloc[3] = ls[ii][i]
            df.iloc[4] = pmd[ii][i]
            df.iloc[5] = pc[ii][i]
            df.iloc[6] = ch[ii][i]
            df.iloc[7] = pk[ii][i]
            df.iloc[8] = aoo[ii][i]
            df.insert(0, "Index", ["MEI", "ET", "TF", "LS", "PMD", "PC", "CH", "PK", "AOO"])
            df.to_csv(country + "/Simulation Family " + str(aging[ii]) + " Years Later For " + str(year[i]) + " Years Afterwards.csv", index=False)    
            list1, list2, propretention, avgretention, averagescore, eval_fail = Main(animal, country + "/Simulation Family " + str(aging[ii]) + " Years Later For " + 
                                                                           str(year[i]) + " Years Afterwards.csv", country + " " + str(aging[ii]) + " Years Later For " + 
                                                                           str(year[i]) + " Years Afterwards.csv", death[ii][i], list1, list2, age)
            
            aging_list.append(aging[ii])
            year_list.append(year[i])
            propretention_list.append(propretention)
            avgretention_list.append(avgretention)
            averagescore_list.append(averagescore)
            eval_fail_list.append(eval_fail)
    
    # Save extra information
    df1 = {
        "Aging" : aging_list,
        "Years After" : year_list,
        "Proportion Retention" : propretention_list,
        "Average Retentoion" : avgretention_list,
        "Average Evaluation Score" : averagescore_list,
        "Evaluation Fail Count" : eval_fail_list
    }
    df1 = pd.DataFrame(df1)
    df1.to_csv(animal + "/" + animal + " Information for " + country + ".csv")

# Simulate for every animal
def simulate(num, country, aging = [0], year = [0]):
    Create_Fam(num, country, "Cat", aging, year)
    Create_Fam(num, country, "Dog", aging, year)
    Create_Fam(num, country, "Horse", aging, year)
    Create_Fam(num, country, "Snake", aging, year)
    Create_Fam(num, country, "Turtle", aging, year)

# Simulate for every country
def full_sent(num, aging = [0], year = [0]):    
    simulate(num, "US", aging, year)
    simulate(num, "Brazil", aging, year)
    simulate(num, "Ethiopia", aging, year)

# Simulate
full_sent(100, [0, 5, 10, 15], [0, 5, 10, 15])

# End Time Counter
end_time = time.time()

# Print Time of Program
print("Time to execute: " + str((end_time - start_time)))

