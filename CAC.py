import numpy as np


#gets the cell's 2 neighbors 
def getNeighbors(i,state):
	stateExt = np.concatenate(([0],state,[0]))
	neighbors = stateExt[i:(i+3)]
	return neighbors


#converts a binary array to the int equivalence 
def arrayToInt(bitlist):
	out = 0
	for bit in bitlist:
		out = (out << 1) | int(bit)
	return out	


#
def transition(i,m,state):
	row = i
	col = arrayToInt(getNeighbors(i,state))
	return m[row][col]



def nextState(state,m):
	next = np.zeros(len(state)-1)
	for i in range(0,len(state)-1):
		next[i] = transition(i,m,state)
	return next



def sgn(x):
	if x > 0:
		return 1
	elif x == 0:
		return 0
	else:
		return -1





#returs the sum of the equivalent  
def cellSumatory(m1,m2,state):
	total = 0
	for i in range(0,len(state)-1):
		total += transition(i,m1,state) - transition(i,m2,state)
	return total


#takes a state (line of data) and returns the classification
def getClassification(m1,m2,state):
	return sgn(cellSumatory(m1,m2,state))


    
    
#performs the test and returns accuracy
def test_star(m1,m2,test_set):
	
	tp = 0
	fp = 0
	for t in test_set:
		classification = getClassification(m1,m2,t)
		if classification >= 0:
			if t[-1] == 1:
				tp += 1
			else:
				fp += 1
		else:
			if t[-1] == 0:
				tp += 1
			else:
				fp += 1
	return (tp)/(tp+fp)




#creates classifier            
def create_Ms(train_set):
    n = len(train_set.iloc[0])-1 
    m1 = np.zeros(shape=(n,8))
    m2 = np.zeros(shape=(n,8))
    p=0
    N=0
    print("learning  started")
    for i in range(0,len(train_set)):
        for j in range(0,len(train_set.iloc[i])-1):
            row = j
            col = arrayToInt(getNeighbors(j,train_set.iloc[i]))
            if train_set.iloc[i][n-1] == 1:
                m1[row][col] += 1
                p+=1
            else:
                m2[row][col] += 1
                N+=1
    m1= np.divide(m1,p) #normalization step (p number of positive class objects)
    m2 = np.divide(m2,N)    
    return m1,m2
		
        
    


