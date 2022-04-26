#TODO Idea is accordig to PatchMatch

import os
import sys
print(sys.argv)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'# '-1'
#import tensorflow as tf
from multiprocessing import Event, Process, Queue
import multiprocessing
import random
from collections import deque
import pickle
import Additional_Help_Functions as helper
import AdditionalEvolutionaryFunctions as evoplus
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from MultiObjectiveProblems import MultiobjectiveProbIsland_Alternate_Sim
import numpy as np
import json
import numpy
from deap.benchmarks.tools import diversity, convergence, hypervolume
import argparse
from datetime import datetime


###########################Function Dict #######################################
#################################################################################
functions= {
            'mnist':helper.load_mnist,
            'fashion':helper.load_fashion_mnist,
            'rgb': evoplus.RGBCrossover,
            'uniform':tools.cxUniform,
            'onepoint': tools.cxOnePoint,
            'twopoint':tools.cxTwoPoint,
            'partiallyMatched': tools.cxUniformPartialyMatched,
            'simulatedBinary': tools.cxSimulatedBinary,
            'areacx':evoplus.area_crossover,
            'mutUniformInt':tools.mutUniformInt,
            'mutArea':evoplus.mutate_according_surrodings,
            'patchesMut':evoplus.patches_mutate,
            'Alt': MultiobjectiveProbIsland_Alternate_Sim.MyProblem,
            }

############################ArgParser############################################
#################################################################################

parser = argparse.ArgumentParser(description='TargetClass.')
parser.add_argument('type', metavar='ty', type=str, nargs='+',help='use Training data = True else False ')
parser.add_argument('blogs', metavar='size', type=int, nargs='+',help='Only needed if patches is used')
parser.add_argument('target', metavar='target', type=int, nargs='+',help='target Class')
parser.add_argument('dataset', metavar='data', type=str, nargs='+',help='load Function of Dataset')
#parser.add_argument('rgb', metavar='rgb', type=bool, nargs='+',help='load Function of Dataset')
parser.add_argument('model', metavar='model', type=str, nargs='+',help='Path to Model')
parser.add_argument('nbr_classes', metavar='nbr_classes', type=int, nargs='+',help='Number of Classes')
parser.add_argument('problem', metavar='problem', type=str, nargs='+',help='Problem Class')
parser.add_argument('epochs', metavar='epochs', type=int, nargs='+',help='Number of max epochs')
parser.add_argument('population', metavar='population', type=int,default=1000, nargs='+',help='Population Size')
parser.add_argument('migration_rate', metavar='migration', type=int,default=1, nargs='+',help='Migration Rate')
parser.add_argument('crossover', metavar='crossover', type=str, nargs='+',help='Crossover Method')
parser.add_argument('cx', metavar='cx', type=str, nargs='+',help='Crossover Rate')#TODO change dtype
parser.add_argument('mutation', metavar='mutation', type=str, nargs='+',help='Mutation Method')
parser.add_argument('mut', metavar='mut', type=str, nargs='+',help='mutation rate')#TODO change dtype
parser.add_argument('algorithm', metavar='algo', type=str, nargs='+',help='NSGA II or III')
parser.add_argument('save', metavar='save', type=str, nargs='+',help='Path to save results to ')
parser.add_argument('sim', metavar='sim', type=str, nargs='+',help='Path to save results to ')
args = parser.parse_args()
#sys.argv=[]

################################################################################
#################################################################################

path = str(args.save[0]) + str(args.target[0])
if not os.path.exists(path):
    os.makedirs(path)

with open(path+'/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
#print('!!!!!!!!' , args)
#################################################################################
#################################################################################

MU=args.population[0]
MIG_RATE = args.migration_rate[0]
NGEN= args.epochs[0]
blogs=args.blogs[0]
print(blogs)

#######################Parameters################################################
#################################################################################


#TODO Fix Mutation Probability and Imdpb
# INDPB independet probability for each attribute to be mutated
# MUTPB:

##################################################################################
##################################################################################

stoplog = numpy.zeros(args.nbr_classes[0], dtype=bool)
func=functions[args.dataset[0]]
trainX, trainY, testX, testY = func()
print(testX[0].shape)
MyProblem=functions[args.problem[0]]
print('Problem instantiated')
equation_inputs = np.around(testX[args.target[0]])
equation_inputs=equation_inputs.astype(int)
#print(np.around(equation_inputs))
shape=equation_inputs.shape
print('Shape', shape)
IND_SIZE = len(equation_inputs.reshape(-1))
print('IND_SIZE',IND_SIZE)
path_model= args.model[0]

toolbox = base.Toolbox()
# Attribute generator
mate=functions[args.crossover[0]]

if args.crossover[0] in ['onepoint', 'twopoint']:
    toolbox.register("mate", mate)
else:
    toolbox.register("mate",mate , indpb=0.5)#args.cx[0])#0.5)

if args.mutation[0]=='mutUniformInt':
    toolbox.register("mutate", evoplus.mutUniformInt_wrapper)
    #else:
    #    toolbox.register("mutate", functions[args.mutation[0]], low=0, up=255, indpb=args.mut[0])#0.1)

elif args.mutation[0] == 'randomAugmentedPatch':
    print('Main: Random Patch Mutate')
    #TODO Eliminated indpb
    toolbox.register("mutate", evoplus.random_patch_mutate, original=equation_inputs,blogs= args.blogs[0], shape= shape)  # 0.1)
else:
    toolbox.register("mutate", functions[args.mutation[0]])


def main(procid, pipein, pipeout, sync,l,stoplog,f=None, seed=None):
    if args.cx[0].endswith('auto2'):
        print('Automation Mode 2')
        CXPB = np.random.rand(MU)

    else:
        print('Not Automation Mode')
        CXPB = float(args.cx[0])  # 0.5

    if args.mut[0].endswith('auto2'):
        MUTPB = np.random.rand(MU)
    else:
        print('Not Automation Mode')
        MUTPB = float(args.mut[0])
    f= open(path+'/'+'hypervolume'+str(procid)+'.csv','w')
    f.write('gen,deme,hypervolume,diversity,convergence \n')
    import tensorflow as tf
    random.seed(seed)
    print('MAIN', procid)
    #l.acquire()
    #sync.set()
    cnn=tf.keras.models.load_model(path_model)
    #l.release()
    print('Model is loaded')
    input_group = cnn.predict(equation_inputs.reshape(1, shape[0], shape[1], shape[2])/255)
    input_group = input_group.argmax()
    print('input group')
    target = procid
    if (stoplog[0] == -1):
        target = random.randint(0, args.nbr_classes[0])
    if (procid >= input_group):
        target=procid+1
    if args.problem[0]=='Alt':
        problem = MyProblem(model=cnn, equation_inputs=equation_inputs, input_group=input_group, procid=target, measure=args.sim[0])
    else:
        problem = MyProblem(model=cnn, equation_inputs=equation_inputs, input_group=input_group, procid=target)
    #print(problem)
    num_weights = problem.number_objectives()
    weights = np.ones(num_weights) * -1
    creator.create("FitnessMin", base.Fitness, weights=weights.tolist())
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("migrate", evoplus.migPipe, k=5, pipein=pipein, pipeout=pipeout,
                     selection=tools.selBest, replacement=random.sample)
    # Structure initializers
    toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
    toolbox.register("select", tools.selNSGA2)
    if(args.type[0]=='True'):
        print('Use Training Data')
        toolbox.register("attribute", random.randint, 0, 255)
        toolbox.register("random_sampling", helper.randomImage, trainX,trainY,target)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.random_sampling)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    elif (args.type[0]=='Single'):
        print('Input Picture')
        toolbox.register("single_image",helper.singleImage,equation_inputs )
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.single_image)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    elif (args.type[0]=='Augmented'):
        print('Input Augmented Picture')
        toolbox.register("data_augmentation",helper.data_augmentation,equation_inputs, shape )
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.data_augmentation)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    elif (args.type[0]=='Patches'):
        print('Input Augmented Patches')
        toolbox.register("data_augmentation",helper.data_augmentation_patch_based,equation_inputs, shape,blogs=blogs,data_augmentation=True)#, max_patches= 400 )
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.data_augmentation)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    elif (args.type[0]=='ShufflePatchesAugmented'):
        print('Input Augmented Patches')
        toolbox.register("data_augmentation",helper.data_augmentation_patch_based,equation_inputs, shape,shuffle=True,blogs=blogs,data_augmentation=True )
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.data_augmentation)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    elif (args.type[0] == 'ShufflePatches'):
        print('Input Augmented Patches')
        toolbox.register("data_augmentation", helper.data_augmentation_patch_based, equation_inputs, shape,data_augmentation=False,shuffle=True,blogs=blogs)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.data_augmentation)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    else:
        print('Use no additional data')
        toolbox.register("attribute", random.randint, 0, 255)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    deme = toolbox.population(n=MU)

    if args.mutation[0] == 'randomPatch':
        toolbox.register("mutate", evoplus.random_patch_mutate, patch_repo=deme)
    elif args.mutation[0] == 'randomAugmentedPatch':
        print('Main: Random Patch Mutate')
        toolbox.register("mutate", evoplus.random_patch_mutate, original=equation_inputs, blogs=args.blogs[0],
                         shape=shape)
    paretohof=tools.ParetoFront(similar=evoplus.pareto_eq)
    hof= tools.HallOfFame(1,similar=evoplus.pareto_eq)
    logbook, mstats=problem.logging(tools)
    l.acquire()
    sync.set()
    for ind in deme:
        i=ind
        ind = helper.reshape_to_image(np.array(ind), shape=shape)
        i.fitness.values = toolbox.evaluate(ind.reshape(-1), target)
    deme = toolbox.select(deme, MU)
    l.release()
    record = mstats.compile(deme)
    logbook.record(gen=0, deme=target, evals=len(deme), **record)
    if procid == 0:
        # Synchronization needed to log header on top and only once
        print(logbook.stream)
        sync.set()
    else:
        logbook.log_header = False  # Never output the header
        sync.wait()
        print(logbook.stream)

    gen = 1

    #TODO Flex
    while gen < NGEN:
        stopping = 0
        for ind in deme:
            x=np.array(ind)
            if args.type[0] == 'Patches' or args.type[0] == 'ShufflePatches' :
                x = helper.reshape_to_image(np.array(ind), shape=shape)
                x=np.array(x,dtype=int)
                #check_image(x, procid, 'Before Iteration')
                #print('done')
            #TODO devision ?
            ind.output=cnn.predict(x.reshape(1,shape[0],shape[1],shape[2])/255)


        paretohof.update(deme)
        hof.update(deme)

        offspring = tools.selTournamentDCD(deme, len(deme))
        print('Start Algo')

        if args.mut[0]=='auto2':
            print('Special Adapt')
            offspring,CXPB,MUTPB=evoplus.adaptVarAnd(offspring,toolbox,CXPB,MUTPB, shape)
        else:
            print('Normal')
            offspring= algorithms.varAnd(offspring, toolbox, cxpb=CXPB, mutpb=MUTPB)
        print('End Algo')

        if MIG_RATE>0:
            if gen % MIG_RATE == 0 and gen > 0:
                toolbox.migrate(offspring)
        print (1)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(2)

        pic = [helper.reshape_to_image(np.array(ind), shape).reshape(-1) for
                   ind in invalid_ind]

        print(3)
        fitnesses = toolbox.map(toolbox.evaluate, pic)
        print(4)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(5)
        deme = toolbox.select(deme + offspring, MU)
        print(6)
        print(7)
        mi= np.zeros(num_weights)
        ma=np.ones(num_weights)
        """Logging Section"""
        if f != None:
            print('File Writer')
            f.write(str(gen) + ',' + str(target) + ',' + str(hypervolume(deme, ma)) + ',' + str(
                    diversity(deme, mi, ma)) + ',' + str(convergence(deme, [mi])) + '\n')
            print('End Filewriter')
        record = mstats.compile(deme)
        print(8)
        logbook.record(gen=gen, deme=target, evals=len(deme), **record)
        print(9)
        print(logbook.stream)
        #####################Stop Criterium ############################
        try:
            if hypervolume(deme, ma) >= 0.9:
                stopping = 1
                print(target, ' stopped')
            stoplog[procid] = stopping
            check = np.array(stoplog)
            print(check)
            if check.sum() == len(check):
                print('ALL STOPPED')
                gen = NGEN
            else:
                print('gent+1')
                gen = gen + 1
        except e:
            print('WHY DO YOU STOOOOP')
            print(e)
        ##############################################################
    print('Start Parteo Dump')
    #if target != input_group:
    try:
        print('a')
        pickle.dump(paretohof, open(str(args.save[0]) + str(args.target[0]) + '/' + "paretoInd_" + str(target) + ".pkl", "wb"), -1)
        pickle.dump(hof,
                    open(str(args.save[0]) + str(args.target[0]) + '/' + "BestHofInd_" + str(target) + ".pkl", "wb"), -1)
        print('b')
    except e:
        print('Error when saving data')
        print(e)
    try:
        print('c')
        pickle.dump(logbook, open(str(args.save[0]) + str(args.target[0]) + '/' + "fitness_" + str(target) + ".pkl", "wb"))
        print('d')
    except e:

        print('Error when saving data')
        print(e)
    try:
        print('e')
        hy=hypervolume(deme, ma)
        print('f')
        print("Final population hypervolume is %f" % hy)
    except e:
        print('Error when saving data')
        print(e)



if __name__ == "__main__":
    start = datetime.now()

    pickle.dump(equation_inputs,
                open(str(args.save[0])+ str(args.target[0]) + '/' + 'beginningIm' + str(np.argmax(testY[args.target[0]])) + '.pkl',
                     'wb'))
    random.seed(64)
    NBR_DEMES = args.nbr_classes[0]-1

    stoplog = multiprocessing.Array('i', NBR_DEMES)
    l = multiprocessing.RLock()
    pipes = [Queue(False) for _ in range(NBR_DEMES)]
 
    for p in pipes:
        print(p)
    pipes_in = deque(p for p in pipes)
    pipes_out = deque(p for p in pipes)
    pipes_in.rotate(1)
    pipes_out.rotate(-1)
    e = Event()
    if NBR_DEMES==0:
        NBR_DEMES = 1
        stoplog = multiprocessing.Array('i', NBR_DEMES)
        stoplog[0]=-1
        l = multiprocessing.RLock()
        pipes = [Queue(False) for _ in range(NBR_DEMES)]
        pipes_in = deque(p for p in pipes)
        pipes_out = deque(p for p in pipes)
        pipes_in.rotate(1)
        pipes_out.rotate(-1)
        processes = [Process(target=main,args=(i, ipipe, opipe, e,l,stoplog, random.random())) for i, (ipipe, opipe) in
                 enumerate(zip(pipes_in, pipes_out))]
    else:
        processes = [Process(target=main, args=(i, ipipe, opipe, e, l, stoplog,random.random())) for
                     i, (ipipe, opipe) in
                     enumerate(zip(pipes_in, pipes_out))]

    for proc in processes:
        print('start')
        proc.start()

    for proc in processes:
        print('join')
        proc.join()

    end = datetime.now()
    f = open(str(args.save[0])+ str(args.target[0]) + '/' + 'time.txt', "w")
    f.write(str((end-start)))
    f.close()


    #f.close()