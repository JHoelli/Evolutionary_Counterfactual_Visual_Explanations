import random
import numpy as np
from Additional_Help_Functions import data_augmentation


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

def Steffen_mutate(individual):
    '''add row at top or bottom '''
    size = len(individual)
    dim =int(np.sqrt(size))
    #individual=np.array(individual)
    #individual=individual.reshape(dim,dim)
    if random.random() < 0.5:
        for i in range(0,size-dim):
            individual[i+dim]=individual[i]

        for j in range(0,dim):
            individual[j]=1
    else:
        for i in range(0,size-dim):
            individual[size-dim-2]=individual[size-1]
        for j in range(0,dim):
            individual[size-j-1]=1
        #individual[27]=np.ones(28)

    #individual=individual.reshape(-1)

    #a=individual
    return individual,

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

def mutate_patches():
    return 'To impelement'