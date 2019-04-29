

import pandas as pd
import numpy as np 
import random

def read_data(datapath):
    """
    This Function reads data and deletes colums with only 0 or only 1 values
    datapath = str(datapath)
    return: data = pd.dataframe              
    """
    data = pd.read_csv(datapath, index_col='ID')
    ####Deleting cols. with only 0 or only 1 values: 
    df = (data.sum (axis = 0) != 0) & (data.sum (axis = 0) != len(data)) 
    data = data[df[df].index]
    return data


def initialization(data, PoN):
    """
    This Function initilizes a Population in which each individual consists of 6 Genes
    data = pd.dataframe
    PoN = population size
    return: biny = np.array(np.array) of binarized expression levels
            mirna = list(list) of miRNAs
    """ 
    biny = []
    mirna = []
    for i in range (PoN):
        #sample for each individual 6 miRNAs
        indi = data.sample(n=6, axis =1)
        indi_b = indi.values.T
        indi_m = list(indi.columns)
        biny.append(indi_b)
        mirna.append(indi_m)
        #for each miRNA 0.5 percent chance of not-transformation
        for j in range(len(biny[i])):
            if random.uniform(0,1)< 0.5:
                biny[i][j] = 1 - biny[i][j]
                mirna[i][j] = "not_" + mirna[i][j]
    return biny, mirna



def fitness(biny):
    """
	This Function calculates Fitness value for each Individuum in Population
	pop: list(dict)(population)
	return: fit =  list(Fitness values) 
	"""
    fit = []
    for i in range(len(biny)):
        #for each individual create confusion tuple
        confused = (0,0,0,0)
        for j in range(len(biny[i])):
            #calculate sum of confusion of one gen an total confusion of individual
            confused = tuple(map(sum, zip(confused, eva(biny[i][j]))))
        #create fitness value for each individual with kappa coefficent
        fit.append(kappa(confused))
    return fit



def eva(arr):
    """
	this function evaluates array of one miRNA in one Individuum
	arr: biny[i][j] np.array
	return: tuple(tn, fp, tp, fn)
	"""
    tn = 0
    fp = 0
    tp = 0
    fn = 0
    for i in range(0, 11):
        if (arr[i] == 0):
            tn +=1
        else:
            fp +=1
    for i in range(11, len(arr)):
        if (arr[i]==1):
            tp +=1
        else:
            fn +=1
    return (tn, fp, tp, fn)


def kappa(t):
    '''
    This Function calculates the Kappa Corellation Coefficent for a Confusion Matrix. 
    Kappa Corellation Coefficient is good for imbalanced data. Range (0,1)
    t: tuple of tn, fp, tp, fn
    return: kappa value
    '''
    cancer = 167
    healthy= 11
    total = t[0]+t[3]+t[1]+t[2]
    Obs_acc = (t[2]+t[0])/total
    Exp_acc = (((cancer*(t[2]+t[3]))/total)+((healthy*(t[0]+t[1]))/total))/total
    Kappa = (Obs_acc-Exp_acc)/(1-Exp_acc)
    return Kappa


def turnament(fit,t_size):
    """
    this function plays turnaments with lists
    fit: list(Fitness values)
    return: tuple(indexes of "winners")
    """
    x = random.sample(range(0,len(fit)-1), t_size)
    sublist = sorted([fit[i] for i in x])[t_size-2:]
    return tuple([fit.index(i) for i in sublist])



def crossover(biny, mirna, cp, t_size):
    """
    This Function selects parents through turnament selection and with crossover propability: cp > random int(0,1) does a crossover of  two parents
    biny = np.array(np.array) of binarized expression levels
    mirna = list(list) of miRNAs
    cp = crossover probability
    t_size = size of group for turnament
    return: biny_new, mirna_new of sames size as biny,mirna
    """
    biny_parents = []
    mirna_parents = []
    biny_new = [] 
    mirna_new = [] 
    
    #Selection of Parents for Crossover
    while len(biny_parents)< len(biny):
        w = turnament(fitness(biny), t_size)
        for i in (0,1):
            biny_parents.append(biny[w[i]])
            mirna_parents.append(mirna[w[i]])
    
    while len(biny_new)< len(biny):
        #randomly select two parents for crossover
        x = random.sample(range(0,len(biny_parents)),2)
        b_parent1 = biny_parents[x[0]]
        b_parent2 = biny_parents[x[1]]
        m_parent1 = mirna_parents[x[0]]
        m_parent2 = mirna_parents[x[1]]
        if random.uniform(0,1) < cp:
            #initilize new individuums, same size as parent individuums
            b_new1, b_new2 = np.zeros_like(biny[0]), np.zeros_like(biny[0])
            m_new1, m_new2 = ["a"]*6, ["a"]*6
            #start crossover
            for i in range(len(biny[i])):
                #for each gene if random(0,1)<0.5 exchange entrys
                if random.uniform(0,1)< 0.5:
                    b_new1[i] = b_parent2[i]
                    b_new2[i] = b_parent1[i]
                    m_new1[i] = m_parent2[i]
                    m_new2[i]= m_parent1[i]

                else:
                    b_new1[i] = b_parent1[i]
                    b_new2[i] = b_parent2[i]
                    m_new1[i] = m_parent1[i]
                    m_new2[i]= m_parent2[i]
                    
            biny_new.append(b_new1)
            biny_new.append(b_new2)
            mirna_new.append(m_new1)
            mirna_new.append(m_new2)
        
        else:
            biny_new.append(b_parent1)
            biny_new.append(b_parent2)
            mirna_new.append(m_parent1)
            mirna_new.append(m_parent2)
        #check for deleting old parent out of list of parents so index doesnt exed matrix dimension
        if x[0] < x[1]:
            biny_parents.pop(x[1])
            biny_parents.pop(x[0])
            mirna_parents.pop(x[1])
            mirna_parents.pop(x[0])
        else:
            biny_parents.pop(x[0])
            biny_parents.pop(x[1])
            mirna_parents.pop(x[0])
            mirna_parents.pop(x[1])
    return biny_new, mirna_new



def mutation(biny,mirna,mp, data):
    """
    This function mutates a individual in the population with mutationprobability mp. 50% change of miRNA 50% change of not-transformation state
    biny = np.array(np.array) of binarized expression levels
    mirna = list(list) of miRNAs
    mp = mutation probability
    data = pd.dataframe
    return biny, mirna
    """
    for i in range(0, len(biny)):
        #for each individual check if random number < cp
        if random.uniform(0,1)< mp:
            #if yes choose random gen for mutation
            zufall = random.randint(0,len(biny[i])-1)
            #if random <0.5 exchange of gene
            if random.uniform(0,1)< 0.5:
                biny[i] = np.delete(biny[i],zufall,axis = 0)
                mirna[i].pop(zufall)
                new = data.sample(n=1, axis =1)
                biny[i] = np.append(biny[i], np.array(new.values.T), axis = 0)
                mirna[i]+=list(new.columns)
            #else: not-transform given gene
            else:
                biny[i][zufall] = 1 - biny[i][zufall]
                if "not" not in mirna[i][zufall]:
                    mirna[i][zufall] = "not_" + mirna[i][zufall]

    return biny, mirna



def main(datapath,pon,t_size,mp,cp,its):
    """
    This function calculates a best classifier from given input data.
    datapath = datapath
    pon = population size
    t_size = turnament size 
    mp = mutation probability
    cp = crossover proability
    its = iterations
    return: best classifier: list(list(mirnas), fitness)
    """
    pop =[]
    data = read_data(datapath) 
    pop = initialization(data,pon)
    dc= [["a"],0.0]
    while its > 0:
        DC = []
        print(its)
        its = its-1
        pop = crossover(pop[0],pop[1],cp,t_size)
        pop = mutation(pop[0],pop[1],mp,data)
        biny = pop[0]
        fit = fitness(biny)
        DC.append(pop[1][fit.index(max(fit))])
        DC.append(max(fit))
        if dc[1] > DC[1]:
            dc = dc
        else:
            dc = [DC[0],DC[1]]
    return dc