import matplotlib.pyplot as plt
from deap import base
from deap import creator
import Additional_Help_Functions as helper
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import datetime
import pickle
import matplotlib.pyplot as plt
from run_metrics import Benchmark
import tensorflow as tf
import re
def picture_example(path,distances=['ssim', 'issm', 'fsim', 'rmse','me'], save='image.png'):
    #distances=['ssim','issm','fsim','rmse','me']
    for dis in distances:
        for folder in os.listdir(f'{path}/{dis}'):
            if not folder.startswith('command'):
                for file in os.listdir(f'{path}/{dis}/{folder}'):
                    if file.startswith('Best'):
                        individual = pickle.load(open(f'{path}/{dis}/{folder}/{file}','rb'))
                        #print(individual[0])
                        target = np.argmax(individual[0].output)
                        plt.figure()
                        #print(np.array(individual).shape)
                        plt.imshow(helper.reshape_to_image(np.array(individual[0]), (28,28,1)).reshape(28,28), cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(f'{path}/{dis}/{folder}/CF_{target}.png', transparent=True, bbox_inches='tight')
                        plt.close()
                    if file.startswith('beginning'):
                        original= pickle.load(open(f'{path}/{dis}/{folder}/{file}','rb'))
                        original_class= 0
                        plt.figure()
                        plt.imshow(np.array(original).reshape(28, 28), cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(f'{path}/{dis}/{folder}/Original.png', transparent=True, bbox_inches='tight')
                        plt.close()

def ssim_closest_image(path,distances=['ssim', 'issm', 'fsim', 'rmse','me'], save='image.png', data  = 'MNIST'):
    #distances=['ssim','issm','fsim','rmse','me']
    if data == 'MNIST':
        x_train, y_train, _ ,_= helper.load_mnist()
    elif data == 'Fashion':
        x_train, y_train, _, _ = helper.load_fashion_mnist()

    for dis in distances:
        for folder in os.listdir(f'{path}/{dis}'):
            if not folder.startswith('command'):
                for file in os.listdir(f'{path}/{dis}/{folder}'):
                    if file.startswith('Best'):
                        print(f'{path}/{dis}/{folder}')
                        individual = pickle.load(open(f'{path}/{dis}/{folder}/{file}','rb'))
                        target = np.argmax(individual[0].output)
                        individual=helper.reshape_to_image(np.array(individual[0]), (28,28,1)).reshape(28,28)
                        index=np.where(np.argmax(y_train, axis=1)== target)
                        x = np.take(x_train, index[0],0)
                        #print(x_train.shape)
                        #print(x_train)
                        ssim_min = 784
                        for a in x:
                            s = (1 - ssim(individual.reshape(28, 28), a.reshape(28, 28))) / 2
                            if ssim_min > s:
                                ssim_min = s
                                ssim_pic = a
                        plt.figure()
                        plt.imshow(np.array(ssim_pic).reshape(28, 28), cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(f'{path}/{dis}/{folder}/SSIM_{target}.png', transparent=True, bbox_inches='tight')
                        plt.close()


def picture_example_added_and_deleted_pixels(path,distances=['ssim', 'issm', 'fsim', 'rmse','me'], save='image.png'):
    for dis in distances:
        for folder in os.listdir(f'{path}/{dis}'):
            if not folder.startswith('command'):
                for file in os.listdir(f'{path}/{dis}/{folder}'):
                    if file.startswith('Best'):
                        individual = pickle.load(open(f'{path}/{dis}/{folder}/{file}','rb'))
                        target = np.argmax(individual[0].output)
                        pic= helper.reshape_to_image(np.array(individual[0]), (28,28,1)).reshape(28,28)
                        cf=pic
                    if file.startswith('beginning'):
                        original= pickle.load(open(f'{path}/{dis}/{folder}/{file}','rb'))
                        original_class= 0

                substract = np.subtract(original.reshape(1, -1), pic.reshape(1, -1))
                substract[np.abs(substract) < 20] = 0
                org = np.array(original).reshape(28, 28)
                substract = substract.reshape(28, 28)
                I = np.dstack([org, org, org])


                orig_delete = np.copy(I)
                orig_add = np.copy(I)

                '''Plot on original'''

                for h in range(substract.shape[0] - 1):  # for every pixel:
                    for j in range(substract.shape[1] - 1):

                        if substract[h, j] > 0:
                            orig_add[h, j] = (255, 0, 0)
                            orig_delete[h, j] = (cf[h,j], cf[h,j], cf[h,j])
                        elif substract[h, j] < 0:
                            orig_delete[h, j] = (0, 255, 0)


                    plt.figure()
                    plt.autoscale(tight='True')
                    plt.imshow(orig_add.reshape(28, 28, 3))
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(f'{path}/{dis}/{folder}/Added_{target}.png',transparent=True, bbox_inches='tight')
                    plt.close()

                    plt.figure()
                    plt.autoscale(tight='True')
                    plt.imshow(orig_delete.reshape(28, 28, 3))
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(f'{path}/{dis}/{folder}/Deleted_{target}.png',transparent=True, bbox_inches='tight')
                    plt.close()





def hypervolume_experiment1(path,distances=['ssim', 'issm', 'fsim', 'rmse','me']):
    #distances=['ssim','issm','fsim','rmse','me']
    full=pd.DataFrame()

    for dis in distances:
        list = []
        if not dis in os.listdir(path):
            for f in os.listdir(f'{path}'):
                if not f.endswith('.csv') and not f.endswith('.png'):
                    for folder in os.listdir(f'{path}/{f}/{dis}'):
                        if not folder.startswith('command'):
                            for file in os.listdir(f'{path}/{f}/{dis}/{folder}'):
                                if file.startswith('hypervolume'):
                                    print(f'{path}/{f}/{dis}/{folder}/{file}')
                                    data= pd.read_csv(f'{path}/{f}/{dis}/{folder}/{file}')
                                    if 'hypervolume' in data.keys().values:
                                        list.append(data['hypervolume'].values[-1])
                                    else:
                                        if type(data['gen'].values[-1])==np.int64: #data['gen'].values[-1].is_integer()
                                            list.append(data['ind'].values[-1])
                                        elif data['gen'].values[-1].is_integer():
                                            list.append(data['ind'].values[-1])
                                        else:
                                            list.append(data['gen'].values[-1])
        else:
            for folder in os.listdir(f'{path}/{dis}'):
                if not folder.startswith('command'):
                    for file in os.listdir(f'{path}/{dis}/{folder}'):
                        if file.startswith('hypervolume'):
                            print(f'{path}/{dis}/{folder}/{file}')
                            data= pd.read_csv(f'{path}/{dis}/{folder}/{file}')
                            if 'hypervolume' in data.keys().values:
                                list.append(data['hypervolume'].values[-1])
                            else:
                                if type(data['gen'].values[-1])==np.int64: #data['gen'].values[-1].is_integer()
                                    list.append(data['ind'].values[-1])
                                elif data['gen'].values[-1].is_integer():
                                    list.append(data['ind'].values[-1])
                                else:
                                    list.append(data['gen'].values[-1])
        print(list)
        list=np.nan_to_num(list)
        da= pd.DataFrame([[np.mean(list),np.std(list)]],columns=[f'{dis}_mean',f'{dis}_std'],)
        full=pd.concat([full,da],ignore_index=True)
    full.to_csv(f'./{path}/hypervolume.csv')


def run_benchmarks(path, model = '../Model/cnn_Fashion_mnist.h5', distance =['ssim', 'issm', 'fsim', 'rmse','me']):
    ml_model = tf.keras.models.load_model(model)
    y=5

    #TODO target does not work here
    for dis in distance:
        outputs = []
        for image_number in os.listdir(f'{path}/{dis}'):
            #print(type(os.listdir(f'{path}/{image_number}')))
            if not image_number.endswith('.txt'):
                counterfactuals=[]
                target=[]
                for a in sorted(os.listdir(f'{path}/{dis}/{image_number}')):
                    if not a.endswith('.txt'):
                        if a.startswith('beginning'):
                            original = pickle.load(open(f'{path}/{dis}/{image_number}/{a}','rb'))
                            print('Shape of Original',original.shape)
                        if a.startswith('Best'):
                            cf=pickle.load(open(f'./{path}/{dis}/{image_number}/{a}','rb')) # removed np.array
                            target=target + [int(re.findall(r"\d", a)[0])] #The bracet not used be here
                            #print(cf.shape)
                            c=helper.reshape_to_image(np.array(cf[0]), (28,28,1)).reshape(-1)
                            # TODO we need .output ! add "new image" to original/ Cast individual and add data
                            c = creator.Individual(c)
                            c.output=cf[0].output
                            c.fitness=cf[0].fitness
                            #cf=c

                            counterfactuals=counterfactuals+[c] # Former: counterfactuals +

                print(len(counterfactuals))
                print(len(counterfactuals[0]))
                benchmark= Benchmark(ml_model,counterfactuals, original,y,target)
                output=benchmark.run_benchmark()
                if len(outputs) <= 1:
                    print('if')
                    outputs = output
                else:
                    print('ELSE')
                    outputs = pd.concat([outputs, output], ignore_index=True)
                outputs.to_csv(f'{path}/{dis}/benchmark_results.csv')

    return None

def average_covered_hypervolume_per_gen_graph(path,distances=['ssim', 'issm', 'fsim', 'rmse','me']):
    '''
    TODO * Make less complicated
        * Check length !
    '''
    #distances = ['ssim', 'issm', 'fsim', 'rmse'] #'EXCLUDED me'
    full = pd.DataFrame()
    #path = path + '/MNIST_final'
    plt.figure()
    for dis in distances:
        list = []
        if not dis in os.listdir(path):
            for f in os.listdir(f'{path}'):
                if not f.endswith('.csv') and not f.endswith('.png'):
                    for folder in os.listdir(f'{path}/{f}/{dis}'):
                        if not folder.startswith('command'):
                            for file in os.listdir(f'{path}/{f}/{dis}/{folder}'):
                                if file.startswith('hypervolume'):
                                    print(f'{path}/{f}/{dis}/{folder}/{file}')
                                    data = pd.read_csv(f'{path}/{f}/{dis}/{folder}/{file}')
                                    if 'hypervolume' in data.keys().values:
                                        if list != []:
                                            list = np.vstack([list, data['hypervolume']])
                                        else:
                                            list = [data['hypervolume']]
                                    else:
                                        if type(data['gen'].values[
                                                    -1]) == np.int64:  # data['gen'].values[-1].is_integer()
                                            if list != []:
                                                list = np.vstack([list, data['ind']])
                                            else:
                                                list = [data['ind']]
                                        elif data['gen'].values[-1].is_integer():
                                            if list != []:
                                                list = np.vstack([list, data['ind']])
                                            else:
                                                list = [data['ind']]

                                        else:
                                            if list != []:
                                                list = np.vstack([list, data['gen']])

                                            else:
                                                list = [data['gen']]
        else:
            for folder in os.listdir(f'{path}/{dis}'):
                if not folder.startswith('command'):
                    for file in os.listdir(f'{path}/{dis}/{folder}'):
                        if file.startswith('hypervolume'):
                            print(f'{path}/{dis}/{folder}/{file}')
                            data= pd.read_csv(f'{path}/{dis}/{folder}/{file}')
                            if 'hypervolume' in data.keys().values:
                                if list != []:
                                    list=np.vstack([list,data['hypervolume']])
                                else:
                                    list= [data['hypervolume']]
                            else:
                                if type(data['gen'].values[-1])==np.int64: #data['gen'].values[-1].is_integer()
                                    if list != []:
                                        list=np.vstack([list,data['ind']])
                                    else:
                                        list = [data['ind']]
                                elif data['gen'].values[-1].is_integer():
                                    if list != []:
                                        list=np.vstack([list,data['ind']])
                                    else:
                                        list = [data['ind']]

                                else:
                                    if list != []:
                                        list=np.vstack([list,data['gen']])

                                    else:
                                        list = [data['gen']]

        print(list)
        list=np.nan_to_num(list)
        print(np.mean(list,axis=1))
        print(np.mean(list, axis=1).shape)
        means=np.mean(list,axis=0)
        stds = np.std(list,axis=0)
        plt.plot(means, label=f'{dis}')
        plt.fill_between(range(99), means - stds, means + stds, alpha=.1)
    plt.ylabel('hypervolume')
    plt.xlabel('epochs')
    plt.legend(loc=4)

    plt.savefig(f'{path}/hypervolume.png')

    #print(list)
    #da= pd.DataFrame([[np.mean(list),np.std(list)]],columns=[f'{dis}_mean',f'{dis}_std'],)
    #full=pd.concat([full,da],ignore_index=True))

    return None
def time(path,distances=['ssim', 'issm', 'fsim', 'rmse','me']):
    #istances = ['ssim', 'issm', 'fsim', 'rmse','me']

    full = pd.DataFrame()

    for dis in distances:
        list = []

        if not dis in os.listdir(path):
            for f in os.listdir(f'{path}'):
                if not f.endswith('.csv') and not f.endswith('.png'):
                    for folder in os.listdir(f'{path}/{f}/{dis}'):
                        if not folder.startswith('command'):
                            for file in os.listdir(f'{path}/{f}/{dis}/{folder}'):
                                if file.startswith('time'):
                                    file = open(f'{path}/{f}/{dis}/{folder}/{file}')
                                    # data = pd.read_csv(f'{path}/{dis}/{folder}/{file}')
                                    # print(data[' gen'].values[-1])
                                    # line=file.readline()
                                    # x = line.split(":")
                                    time = datetime.datetime.strptime(file.readline(), '%H:%M:%S.%f').time()
                                    list.append(time.hour * 60 + time.minute)
                                    print(list)
        else:
            for folder in os.listdir(f'{path}/{dis}'):
                if not folder.startswith('command'):
                    for file in os.listdir(f'{path}/{dis}/{folder}'):
                        if file.startswith('time'):
                            file=open(f'{path}/{dis}/{folder}/{file}')
                            #data = pd.read_csv(f'{path}/{dis}/{folder}/{file}')
                            # print(data[' gen'].values[-1])
                            #line=file.readline()
                            #x = line.split(":")
                            time = datetime.datetime.strptime(file.readline(), '%H:%M:%S.%f').time()
                            list.append(time.hour*60+time.minute)
                            print(list)
        da = pd.DataFrame([[np.mean(list), np.std(list)]], columns=[f'{dis}_mean', f'{dis}_std'], )
        full = pd.concat([full, da], ignore_index=True)
    full.to_csv(f'./{path}/time.csv')


if __name__=='__main__':
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    '''Comparison Only MNIST but 30'''
    #average_covered_hypervolume_per_gen_graph('./Old_Distances/MNIST_final')
    #hypervolume_experiment1('./Old_Distances/MNIST_final')
    #time('./Old_Distances/MNIST_final')
    '''Experiment 1: Dataset specific Results'''
    #MNIST
    average_covered_hypervolume_per_gen_graph('./Distance/MNIST_15')
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    picture_example('./Distance/MNIST_15')
    picture_example_added_and_deleted_pixels('./Distance/MNIST_15')
    ssim_closest_image('./Distance/MNIST_15')
    hypervolume_experiment1('./Distance/MNIST_15')
    time('./Distance/MNIST_15')
    #Fashion
    average_covered_hypervolume_per_gen_graph('./Distance/Fashion_15')
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    picture_example('./Distance/Fashion_15')
    picture_example_added_and_deleted_pixels('./Distance/Fashion_15')
    ssim_closest_image('./Distance/Fashion_15', data='Fashion')
    hypervolume_experiment1('./Distance/Fashion_15')
    time('./Distance/Fashion_15')
    ''' Experiment 1: Combined Results'''
    average_covered_hypervolume_per_gen_graph('./Distance')
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    hypervolume_experiment1('./Distance')
    time('./Distance')
    #In here pic creation not necessary --> is done before

    run_benchmarks('./Distance/MNIST_final', '../Model/cnn_MNIST.h5')
    average_covered_hypervolume_per_gen_graph('./Distance')
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    picture_example('Distance/Fashion_15')
    hypervolume_experiment1('./Distance/MNIST_final')
    time('./Distance/MNIST_final')

    '''Experiement 2: Mutation - MNIST '''
    average_covered_hypervolume_per_gen_graph('./Mutation/MNIST',['mutUniformInt','mutDataAug'])
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    picture_example('./Mutation/MNIST',['mutUniformInt','mutDataAug'])
    picture_example_added_and_deleted_pixels('./Mutation/MNIST',['mutUniformInt','mutDataAug'])
    ssim_closest_image('./Mutation/MNIST',distances=['mutUniformInt','mutDataAug'])
    hypervolume_experiment1('./Mutation/MNIST',['mutUniformInt','mutDataAug'])
    time('./Mutation/MNIST',['mutUniformInt','mutDataAug'])

    '''Experiement 2: Mutation - Fashion '''
    average_covered_hypervolume_per_gen_graph('./Mutation/Fashion_me',['mutUniformInt','mutDataAug'])
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    picture_example('./Mutation/Fashion_me',['mutUniformInt','mutDataAug'])
    picture_example_added_and_deleted_pixels('./Mutation/Fashion_me', ['mutUniformInt', 'mutDataAug'])
    ssim_closest_image('./Mutation/Fashion_me', distances=['mutUniformInt', 'mutDataAug'], data='Fashion')
    hypervolume_experiment1('./Mutation/Fashion_me',['mutUniformInt','mutDataAug'])
    time('./Mutation/Fashion_me',['mutUniformInt','mutDataAug'])

    '''Experiment2: Combined'''
    average_covered_hypervolume_per_gen_graph('./Mutation',['mutUniformInt','mutDataAug'])
    run_benchmarks('./Distance/MNIST_final','../Model/cnn_MNIST.h5')
    hypervolume_experiment1('./Mutation',['mutUniformInt','mutDataAug'])
    time('./Mutation',['mutUniformInt','mutDataAug'])

