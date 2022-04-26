import random

import deap.tools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from Additional_Help_Functions import data_augmentation, data_augmentation_patch_based
from deap import creator,base
from deap.benchmarks.tools import diversity
import Additional_Help_Functions as helper
# Geometric Semantic CX https://kar.kent.ac.uk/69663/1/semantic-gp.pdf

def augmented_mutate(ind1,shape=None):
    ind1=np.array(ind1)
    #print('IND1', ind1.shape)
    ind1 =ind1.reshape(shape[0],shape[1],shape[2])
    ind1= data_augmentation(ind1, shape=shape)
    ind1 = ind1.reshape(-1)
    ind1=creator.Individual(ind1)
    return ind1,

def mutUniformInt_wrapper(ind1):
    ind1 = np.array(ind1)
    shape= ind1.shape
    ind1 = deap.tools.mutUniformInt(ind1.reshape(-1),0,255,0.1)
    ind1 = np.array(ind1).reshape(shape[0], shape[1], shape[2],shape[3])
    ind1 = creator.Individual(ind1)
    return ind1,

def adapt_dong_cross(f_a,f_b,mi,ma):
    '''Implemented According to https://link.springer.com/content/pdf/10.1007%2F978-3-642-05253-8_16.pdf'''
    cross = abs(f_a-f_b)/(ma-mi)
    return cross
def adapt_dong_mut(f_a, f_b, mi, ma):
    mut = 0.2* np.power((f_a)/ma,2)
    return mut
def adapt_fitness(f,f_larger,fmax,favg):
    '''Implemented after https://iopscience.iop.org/article/10.1088/1757-899X/782/4/042028/pdf
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=286385
    '''
    if (f_larger >favg):
        cross=0.9-(0.3*(fmax-f_larger))/(fmax-favg)

        mut = 0.1-((0.1-0.001)*(fmax-f))/(fmax-favg)
    else:
        cross=0.9
        mut=0.1
    return cross,mut

def adapt_diversity(diversity,f,fmin,fmax):
    '''This is implemented according to McGinely2008'''
    cross=adaptive_crossover(diversity)
    return cross#, mut

def adaptive_mutation_individual(diversity,f,fmin,fmax, k=0.2):
    '''This is implemented according to McGinely2008'''
    mut_div= -(diversity-100)/100 *k

    if fmin==fmax:
        mut_fit =0.2
    else:
        mut_fit=-(((f-fmin)/(fmax-fmin)*k)-k)

    mut = (mut_div+mut_fit)/2
    return mut

def adaptive_mutation(diversity, k=0.2):
    '''This is implemented according to McGinely2008
    excluded div
    '''
    mut_div= -(diversity-100)/100 *k
    return mut_div

def adaptive_crossover(scaler, ma=0.8 , mi=0.4):
    '''This is implemented according to McGinely2008'''
    cross = ((scaler/100*(ma-mi))+mi)
    if np.isnan(cross):
        print('CX IS NAN ')
        print(scaler)
    return cross
def adaptVarAnd(population, toolbox,p_c,p_m,shape,all = True):
    '''Taken From Here: https://www.researchgate.net/publication/283339697_Self-tuning_geometric_semantic_Genetic_Programming'''
    print('AdaptVarAnd')
    offspring = [toolbox.clone(ind) for ind in population]
    old=[toolbox.clone(ind) for ind in population]
    for i in range(1, len(offspring), 2):
        p_cross=(p_c[i-1]+p_c[i])/2
        if random.random()<p_cross:
            old1 =offspring[i - 1]
            old2=offspring[i]
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                      offspring[i])

            of1= helper.reshape_to_image(np.array(offspring[i - 1]),shape=shape)
            of1_val=toolbox.evaluate(of1.reshape(-1))
            of2 = helper.reshape_to_image(np.array(offspring[i]), shape=shape)
            of2_val = toolbox.evaluate(of2.reshape(-1))
            old[i].fitness.values=of2_val
            old[i-1].fitness.values=of1_val

            if all:
                of1_val=(of1_val[0]+of1_val[1]+of1_val[2])/3
                of2_val = (of2_val[0] + of2_val[1] + of2_val[2]) / 3
                ol1=(old1.fitness.values[0]+old1.fitness.values[1]+old1.fitness.values[2])/3
                ol2=(old2.fitness.values[0]+old2.fitness.values[1]+old2.fitness.values[2])/3
            else:
                of1_val=of1_val[0]
                of2_val = of2_val[0]
                ol1=old1.fitness.values[0]
                ol2=old2.fitness.values[0]
            if of1_val>ol1 and of1_val>ol2:
                p_c[i-1]=p_cross+0.01
            else:
                p_c[i - 1] = p_cross - 0.01
            if of2_val>ol1 and of2_val>ol2 :
                p_c[i]=p_cross+0.01
            else:
                p_c[i] = p_cross - 0.01
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    #TODO save fitness reults to old ! To check actiual improvement
    for i in range(len(offspring)):
        #print('i',i)
        if random.random() < p_m[i]:
            #old=offspring[i]
            offspring[i], = toolbox.mutate(offspring[i])
            #print(np.array(offspring[i]).shape)
            #TODO array , if !
            of=helper.reshape_to_image(np.array(offspring[i]), shape=shape)
            of=toolbox.evaluate(of.reshape(-1))
            if all:
                of=(of[0]+of[1]+of[2])/3
                #print(old[i])
                #print(old[i].fitness.values)
                ol=(old[i].fitness.values[0]+old[i].fitness.values[1]+old[i].fitness.values[2])/3
            else:
                of=of[0]
                ol=old[i].fitness.values[0]
            #print(toolbox.evaluate(of.reshape(-1)))
            #print(old.fitness.values)
            if of>ol:
                p_m[i]=p_m[i]+0.01
            else:
                p_m[i] = p_m[i] - 0.01
            del offspring[i].fitness.values
    return offspring,p_c,p_m

def adaptiveVarAnd(population, toolbox,num_weights,logbook):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    TODO Not finished yet
    """
    mi = np.zeros(num_weights)
    ma = np.ones(num_weights)
    #div = diversity(population, mi, ma)
    #cxpb= adaptive_crossover(div)
    #print('CXPN',cxpb)
    #TODO does this work?
    values= logbook
    #print(values.chapters["distance"].select("min"))
    #TODO Without scalinh
    fmin=(values.chapters["distance"].select("min")[-1]+values.chapters["sparsity"].select("min")[-1]+values.chapters["output"].select("min")[-1])/3
    fmax= (values.chapters["distance"].select("max")[-1]+values.chapters["sparsity"].select("max")[-1]+values.chapters["output"].select("max")[-1])/3
    favg = (values.chapters["distance"].select("avg")[-1]+values.chapters["sparsity"].select("avg")[-1]+values.chapters["output"].select("avg")[-1])/3
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        f1 = (offspring[i].fitness.values[0]+offspring[i].fitness.values[1]+offspring[i].fitness.values[2])/3
        f2 = (offspring[i-1].fitness.values[0] + offspring[i-1].fitness.values[1] + offspring[i-1].fitness.values[2]) / 3
        #if f1>f2:
        #    f_larger=f1
        #    f_minor=f2
        #else:
        #    f_larger = f2
        #    f_minor = f1
        cxpb=adapt_dong_cross(f1,f2,fmax,favg)
        if random.random() < cxpb:
            #print('mate')
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            #print('mate finsihed')
            #TODO Pt delete back
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    for i in range(len(offspring)):
        #print(offspring[i].fitness)
        #print(offspring[i].fitness.values[0])
        #print(offspring[i].fitness.values[1])
        #print(offspring[i].fitness.values[0])
        f = (offspring[i].fitness.values[0]+offspring[i].fitness.values[1]+offspring[i].fitness.values[2])/3
        mutpb=adapt_dong_mut(f,None,fmax,fmin)

        #mutpb = adaptive_mutation_individual(div,f,fmin,fmax)

        #print('MUTPB', mutpb)
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring

def random_patch_mutate(ind1,original,shape, blogs,indpb=1):
    '''
    Takes original image, does Data Augmentation and devides Image into patches
    :param ind1: individual to mutate
    :param ind2: original image
    :return mutated individual

    TODO
        * Why are patches with doubles
        * How to cope with color
    '''

    # Funtion for only color mutation ?
    #print('Random Patch Mutate')
    #if random.random() < indpb:
    ind1 = np.array(ind1)
    patch_repo= data_augmentation_patch_based(original, shape=shape, data_augmentation = True,shuffle=True, max_patches=None,view_as_blog=True,blogs=blogs)
    #print(patch_repo.shape)
    patch=random.choice(patch_repo)
    #print('patch', patch)
    #print('patch', patch.shape)
    #id=random.choice(patch)
    #print('ID', id)
    #print('ID', id.shape)
    i = random.randint(0,len(ind1)-1)
    #print('i',i)
    ind1[i] = patch
    #print(ind1.shape)
    ind1 = creator.Individual(ind1)
    return ind1,

def migPipe(deme, k, pipein, pipeout, selection, replacement=None):
    """Migration using pipes between initialized processes. It first selects
    *k* individuals from the *deme* and writes them in *pipeout*. Then it
    reads the individuals from *pipein* and replace some individuals in the
    deme. The replacement strategy shall not select twice the same individual.

    :param deme: A list of individuals on which to operate migration.
    :param k: The number of individuals to migrate.
    :param pipein: A :class:`~multiprocessing.Pipe` from which to read
                   immigrants.
    :param pipeout: A :class:`~multiprocessing.Pipe` in which to write
                    emigrants.
    :param selection: The function to use for selecting the emigrants.
    :param replacement: The function to use to select which individuals will
                        be replaced. If :obj:`None` (default) the individuals
                        that leave the population are directly replaced.
    """
    emigrants = selection(deme, k)
    #print(len(emigrants))
    if replacement is None:
        #print('if')
        # If no replacement strategy is selected, replace those who migrate
        immigrants = emigrants
    else:
        #print('else')
        # Else select those who will be replaced
        immigrants = replacement(deme, k)
    #print('start send ')
    pipeout.put(emigrants)
    #print('send finsihed')
    buf = pipein.get()
    #print('rev finsihed')
    for place, immigrant in zip(immigrants, buf):
        indx = deme.index(place)
        deme[indx] = immigrant

def RGBCrossover(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    length = len(ind1)
    row = int(length/3)
    for i in range(row):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
            ind1[row+i], ind2[row+i] = ind2[row+i], ind1[row+i]
            ind1[2*row + i], ind2[2*row + i] = ind2[2*row + i], ind1[2*row + i]


    return ind1, ind2

def RGBMutation():

    return ''

def area_crossover(ind1, ind2,indpb):
    size = min(len(ind1), len(ind2))
    dim= int(np.sqrt(size))
    #save1=ind1
    #save2=ind2
    #ind1=numpy.array(ind1).reshape(dim, dim)
    #ind2 = numpy.array(ind1).reshape(dim, dim)
    x=random.randint(0,dim-2)
    y=random.randint(0,dim-2)
    start= x*(y+1)
    start2= x*(y+2)
    if random.random() < indpb:
        #ind1[x:x+2,y:y+2], ind2[x:x+2,y:y+2] = ind2[x:x+2,y:y+2], ind1[x:x+2,y:y+2]
        ind1[start:start +2],ind2[start:start +2]=ind2[start:start +2],ind1[start:start +2]
        ind1[start2:start2 + 2], ind2[start2:start2 + 2] = ind2[start2:start2 + 2], ind1[start2:start2 + 2]
    #save1[:]=ind1.reshape(-1)
    #print(save1.fitness)
    #save2[:]=ind2.reshape(-1)
    return ind1, ind2

def area_crossover_rgb(ind1, ind2,indpb, shape = (150,150,3)):
    #TODO check if consistent with definition of reshape
    ind1= np.array(ind1)
    ind2=np.array(ind2)
    ind1=ind1.reshape(shape[0],shape[1],shape[2])
    ind2=ind2.reshape(shape[0],shape[1],shape[2])
    #TODO Replace 12 with Patch length
    x=random.randint(0,shape[0]-12)
    y=random.randint(0,shape[1]-12)
    x = random.randint(0, shape[0] -12)
    y = random.randint(0, shape[1] - 12)
    #print(ind1[x:x + 12,y:y+12].shape)
    #ind1 =ind1.tolist()
    #ind2=ind2.tolist()


    ind1[x:x + 12,y:y + 12], ind2[x:x + 12,y:y+ 12] = ind2[x:x + 12,y:y + 12].copy(), ind1[x:x + 12,y:y + 12].copy()
    ind1 = ind1.reshape(-1)
    ind1 = ind1.tolist()

    ind1 = creator.Individual(ind1)
    ind2 = ind2.reshape(-1)
    ind2=ind2.tolist()
    ind2 = creator.Individual(ind2)
    return ind1, ind2

def uniform_crossover_rgb(ind1, ind2,indpb, shape = (150,150,3)):
    #TODO Test
    ind1= np.array(ind1)
    ind2=np.array(ind2)
    ind1=ind1.reshape(shape[0]*shape[1],shape[2])
    ind2=ind2.reshape(shape[0]*shape[1],shape[2])

    ind1 = ind1.tolist()
    ind2 = ind2.tolist()
    for i in range(0,shape[0]*shape[1]):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    ind1=np.array(ind1)
    ind1 = ind1.reshape(-1)
    ind1.tolist()
    ind1 = creator.Individual(ind1)
    ind2 = np.array(ind2)
    ind2 = ind2.reshape(-1)
    ind2=ind2.tolist()
    ind2 = creator.Individual(ind2)

    return ind1, ind2


def patches_mutate(ind):
    '''mutates single value'''
    mutate=[]
    for a in range(0,len(ind)-1):
        pixel= random.randint(0,len(ind[a])-1)
        #print(a)
        #print(pixel)
        #print(ind[a])
        ind[a][pixel]= random.randint(0,255)
        return ind,
def mutate_augmentation_old(ind,shape):
    """selects da Patch and perform Data Augmenatation """
    #TODO IMPLEMENT
    #i = random.randint(0,len(ind[a])-1)
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomContrast(factor=(0.1, 1.3)),
        layers.experimental.preprocessing.RandomZoom(height_factor=(-0.7, -0.2)),
    ])
    # print('START')
    single_image = np.array(ind).reshape(1, shape[0], shape[1], shape[2])
    single_image = single_image.astype(int)
    # seed = (2, 3)
    prob = 0.5
    single_image = data_augmentation(single_image)
    if shape[2] == 3:
        if random.uniform(0, 1) > prob:
            # print('hue')
            single_image = tf.image.random_hue(single_image, 0.08)
        if random.uniform(0, 1) > prob:
            # print('satureation')
            single_image = tf.image.random_saturation(single_image, 0.6, 1.6)
    single_image = np.clip(single_image, 0, 255)
    return ind,

def mutate_according_surrodings(individual):
    '''add row at top or bottom '''
    size = len(individual)
    #print(size)
    dim =int(np.sqrt(size))
    #print(dim)
    #individual=np.array(individual)
    #individual=individual.reshape
    select_Pixel_number=random.randint(0,size-1)
    if select_Pixel_number in [0, dim-1,size-1,size-dim]:
        #print('corner selected Pixel',select_Pixel_number )
        #corner
        surroding = np.zeros(3)
        if(select_Pixel_number==0):
            surroding[0] = individual[select_Pixel_number + 1]
            surroding[1] = individual[select_Pixel_number + dim]
            surroding[2] = individual[select_Pixel_number + 1 + dim]
        elif(select_Pixel_number==dim-1):
            surroding[0] = individual[select_Pixel_number - 1]
            surroding[1] = individual[select_Pixel_number + dim]
            surroding[2] = individual[select_Pixel_number - 1 + dim]
        elif (select_Pixel_number == size-1):
            surroding[0] = individual[select_Pixel_number - 1]
            surroding[1] = individual[select_Pixel_number - dim]
            surroding[2] = individual[select_Pixel_number - 1 - dim]
        elif (select_Pixel_number == dim - dim):
            surroding[0] = individual[select_Pixel_number + 1]
            surroding[1] = individual[select_Pixel_number - dim]
            surroding[2] = individual[select_Pixel_number + 1 - dim]
        else:
            print('')

    elif select_Pixel_number in range(0,dim) :
        #print('First Row selected Pixel', select_Pixel_number)
        #in oberster reihe
        surroding=np.zeros(5)
        surroding[0]=individual[select_Pixel_number-1]
        surroding[1] = individual[select_Pixel_number + 1]
        surroding[2] = individual[select_Pixel_number - 1+dim]
        surroding[3] = individual[select_Pixel_number + 1+dim]
        surroding[4] = individual[select_Pixel_number +dim]
    elif select_Pixel_number in range(size-dim, size-1):
        # in unterste reihe
        #print('Last row selected Pixel', select_Pixel_number)
        surroding = np.zeros(5)
        surroding[0] = individual[select_Pixel_number - 1]
        surroding[1] = individual[select_Pixel_number + 1]
        surroding[2] = individual[select_Pixel_number - 1 - dim]
        surroding[3] = individual[select_Pixel_number + 1 - dim]
        surroding[4] = individual[select_Pixel_number - dim]
    elif select_Pixel_number % dim==0:
        # in links
        #print('left column selected Pixel', select_Pixel_number)
        surroding = np.zeros(5)
        surroding[1] = individual[select_Pixel_number + 1]
        surroding[3] = individual[select_Pixel_number + 1 - dim]
        surroding[4] = individual[select_Pixel_number - dim]
        surroding[0] = individual[select_Pixel_number + dim]
        surroding[2] = individual[select_Pixel_number + dim+1]
    elif select_Pixel_number % dim== (dim-1):
        # in linkf
        #print('right column selected Pixel', select_Pixel_number)
        surroding = np.zeros(5)
        surroding[1] = individual[select_Pixel_number - 1]
        surroding[3] = individual[select_Pixel_number - 1 - dim]
        surroding[4] = individual[select_Pixel_number - dim]
        surroding[0] = individual[select_Pixel_number + dim]
        surroding[2] = individual[select_Pixel_number + dim - 1]
    else:
        #print('Else selected Pixel', select_Pixel_number)
        surroding=np.zeros(8)
        surroding[0]=individual[select_Pixel_number-1]
        surroding[1] = individual[select_Pixel_number + 1]
        surroding[2] = individual[select_Pixel_number - 1-dim]
        surroding[3] = individual[select_Pixel_number + 1-dim]
        surroding[4] = individual[select_Pixel_number -dim]
        surroding[5] = individual[select_Pixel_number +1 +dim]
        surroding[6] = individual[select_Pixel_number - 1+dim]
        surroding[7] = individual[select_Pixel_number +dim]
#    surroding[8] = individual[select_Pixel_number + 1 + dim]
    mi = min(surroding)
    ma= max(surroding)
    individual[select_Pixel_number]=random.randint(mi, ma)
    return individual,

def pareto_eq(ind1, ind2):
        """Determines whether two individuals are equal on the Pareto front

        Parameters
        ----------
        ind1: DEAP individual from the GP population
            First individual to compare
        ind2: DEAP individual from the GP population
            Second individual to compare

        Returns
        ----------
        individuals_equal: bool
            Boolean indicating whether the two individuals are equal on
            the Pareto front

        """
        return np.all(ind1.fitness.values == ind2.fitness.values)
