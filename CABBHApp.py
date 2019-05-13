import math
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime
import CAC
import time


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root





Autism = os.path.join(ROOT_DIR,"Autism_.csv")
Pregnancies = os.path.join(ROOT_DIR,"Pregnancies.csv")
test_file = os.path.join(ROOT_DIR,"test_image.csv")
train_file = os.path.join(ROOT_DIR,"train_image.csv")
heart = os.path.join(ROOT_DIR,"heartB.csv")
diabetes = os.path.join(ROOT_DIR,"diabetesB.csv") 
#test = pd.read_csv(test_file)          #to test this file uncomment these 2 lines and remove lines (205,206,207)
#train = pd.read_csv(train_file)
data = pd.read_csv(diabetes)            #change the parameter to test different dataset
data = data.drop('Unnamed: 0' , axis=1)
feat_num = len(data.iloc[0])
random.seed(datetime.now)

class Star:
    
    def __init__(self,m1,m2,test):
      self.criteria =  (random.uniform(0.3,0.9))  
      self.selection = selection_init()
      self.fitness = star_fitness(self.selection , self.criteria , m1 , m2 ,test)
    
    def set_fitness(self, fit):
        self.fitness = fit
    def get_fitness(self):
        return self.fitness
    def get_star(self , index):
        return self.index(index)
      

#choosing the selection of features
def selection_init ():
        
        arr = []
        for i in range (feat_num-1):
            arr.append(random.random())
        
        return arr

#creating the stars
def stars_init_(stars_num):
    starts_array = []
    for i in range(stars_num):
        starts_array.append(Star(m1 , m2 , test))



#coverts the dataset and m1 and m2  according to the star and calculates fitness
def star_fitness(selection_arr , criteria,m1 , m2 ,test):
    
    binary = mask_converter(selection_arr , criteria)
    mask = np.array(binary)
    m1_masked = m1[mask,:]
    m2_masked = m2[mask,:] 
    
    binary.append(True)
    mask = np.array(binary)
    
  
    masked_test = test.values[:, mask]
    print('the number of selected features :')
    print( len(masked_test[0]))
    acc = CAC.test_star(m1_masked,m2_masked,masked_test)
    print('accuracy = ',acc)
    return  acc
  
    
#choosing the max fitness
def max_fitness(stars):
    max_fit = stars[0].get_fitness()
    max_index = 0
    for star in stars:
        if max_fit < star.get_fitness():
            max_fit = star.get_fitness()
            max_index = stars.index(star)
    print("maximum fitness = " , max_fit)

    return max_index

#returs the number of features in a given star
def feature_count(star):
    count = 0
    for i in star.selection:
        if i > star.criteria:
            count += 1
    return count

#converts the float variables in the selection array to binary array
def mask_converter(selection , criteria):
        binary_arr = []

        for i in selection:
            if(i > criteria):
                binary_arr.append(True)
            else:
                binary_arr.append(False)
        return binary_arr

#calculates the event horizon
def radius(bh_index, stars):
    _sum = 0.0
    for s in stars:
        if stars.index(s) != bh_index:  # is bh included ??
            _sum += s.get_fitness()
           
    return stars[bh_index].get_fitness()/_sum


#the main function ***
def BBHA(stars_num , iterations_number , m1 , m2 , train , test):
    # #create stars#
    list_of_stars = []

    for i in range(stars_num):
        star = Star(m1,m2,test)
        list_of_stars.append(star)


    bh_index = max_fitness(list_of_stars)

    iterations = 0

    #black_hole: Star = copy.copy(list_of_stars[bh_index])
    print ("black hole is the star num" , bh_index)

    # ***** the begining of the loop *********
    while iterations < iterations_number:

        for a in list_of_stars:
            tries = 0
            next_ = False
            #old_fitness = a.get_fitness()
           
            while(next_ == False):
               
                print("Star Num" , list_of_stars.index(a))
                a.set_fitness(star_fitness(a.selection , a.criteria , m1 , m2 ,test))
                if a.get_fitness() > list_of_stars[bh_index].get_fitness():
                          
                          bh_index = list_of_stars.index(a)
                          
                          
                          # remove if not used
                elif a.fitness == list_of_stars[bh_index].fitness and feature_count(list_of_stars[bh_index]) > feature_count(a):
                              
                              bh_index = list_of_stars.index(a)
                              #a.fitness = old_fitness
                _r = radius(bh_index, list_of_stars)
                print ("black hole is the star num" , bh_index , "with fitness" , list_of_stars[bh_index].get_fitness())
                if  math.sqrt(math.pow((list_of_stars[bh_index].fitness - a.fitness), 2)) < _r or tries>=10 :
                                  next_ = True
                                  
                if list_of_stars.index(a) != bh_index:
                    
                    for i  in range (len(a.selection)):
                    
                         a.selection[i] = a.selection[i] + (random.random() * (list_of_stars[bh_index].selection[i] - a.selection[i]))
                         if abs(math.tanh(a.selection[i])) > 0.5:
                                  a.selection[i] = 1
                         else:
                                  a.selection[i] = 0
                tries +=1
        iterations += 1
    print("the number of features selected: ")
    print(feature_count(list_of_stars[bh_index]))

    return star_fitness(list_of_stars[bh_index].selection , list_of_stars[bh_index].criteria , m1 , m2 ,test) ,mask_converter(list_of_stars[bh_index].selection , list_of_stars[bh_index].criteria)


#checks if the column has at least 2 different values (at least 1)
def check_diff(arr):
    stuff = [arr[0]]
    for i in  arr:
        if i  not in stuff:
            return True
    return False


#returns a boolean array where the columns that are the same on all objects are marked false
def remove_repitition(train):
    arr = []
    for i in range(len(train.iloc[0])-1):
        arr.append(check_diff(train.iloc[:,i]))
    arr.append(True)
    return arr



if __name__ == "__main__":
    
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    mask = remove_repitition(train)
    train = pd.DataFrame(train.values[:,mask])
    test = pd.DataFrame(test.values[:,mask])
    feat_num = len(train.iloc[0])
    start = time.time()
    m1,m2 = CAC.create_Ms(train)
    fitt , select = BBHA(10,10,m1,m2,train,test)
    
    
    