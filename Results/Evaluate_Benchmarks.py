
import pandas as pd
import numpy as np
import re
import pickle
from skimage.metrics import structural_similarity as ssim
from deap import base
import pandas as pd
from deap import creator
import random
import os
import  matplotlib
import matplotlib.pyplot as plt
import Additional_Help_Functions as helper
from Utils import getOverallBestPics
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }


def box_plot_distance(path1, path2, path3=None, save = None):
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    d_1 = data1['Distance_1']
    d_2 =data2['Distance_1']
    d_3 = data3['Distance_1']
    plt.boxplot(x=[d_1,d_2,d_3])#,y=['Our Approach','Wachter et al'])#, label='our approach')
    plt.xticks([1, 2, 3], ['Our Approach', 'Wachter et al.', 'Van Looveren et al.'])
    plt.ylim([0, 1])
    plt.savefig(f'./Benchmark/L1_dist_{save}.png',transparent=True, bbox_inches='tight' )
    plt.close()
    plt.boxplot(x=[data1['Distance_2'],data2['Distance_2'], data3['Distance_2']])#,y=['Our Approach','Wachter et al'])#).values,label='our approach')
    plt.xticks([1, 2, 3], ['Our Approach', 'Wachter et al.', 'Van Looveren et al.'])
    plt.ylim([0, 1])
    #plt.boxplot(x=)#.values,label='Wachter et al' )
    plt.savefig(f'./Benchmark/L2_dist_{save}.png', transparent=True, bbox_inches='tight')
    plt.close()



def box_plot_distance_violin(path1, path2, path3=None, save = None):
    import seaborn as sns
    #sns.set_style('whitegrid')
    #ax = sns.violinplot(x='Survived', y='Age', data=df)
    #ax = sns.stripplot(x="Survived", y="Age", data=df)
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 14,
            }
    matplotlib.rc('font', size=14)
    matplotlib.rc('axes', titlesize=14)

    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    d_1 = data1['Distance_1']
    d_2 = data2['Distance_1']
    d_3 = data3['Distance_1']

    plt.violinplot([d_1, d_2, d_3],[1, 2, 3],points=80, vert=True, widths=0.7,
                         showmeans=True, showextrema=True, showmedians=False)
    #axs[1, 0].set_title('Custom violinplot 6', fontsize=fs)

    #plt.boxplot(x=[d_1, d_2, d_3])  # ,y=['Our Approach','Wachter et al'])#, label='our approach')
    plt.xticks([1, 2, 3], ['Our Approach', 'Wachter et al.', 'Van Looveren et al.'],fontsize=14)
    plt.ylim([0, 1])
    plt.savefig(f'./Benchmark/L1_dist_violin_{save}.png', transparent=True, bbox_inches='tight')
    plt.close()

    plt.violinplot([data1['Distance_2'], data2['Distance_2'],
                   data3['Distance_2']],[1, 2, 3],points=80, vert=True, widths=0.7,
                   showmeans=True, showextrema=True, showmedians=False)
    plt.xticks([1, 2, 3], ['Our Approach', 'Wachter et al.', 'Van Looveren et al.'],fontsize=14)
    plt.ylim([0, 1])
    # plt.boxplot(x=)#.values,label='Wachter et al' )
    plt.savefig(f'./Benchmark/L2_dist_violin_{save}.png', transparent=True, bbox_inches='tight')
    plt.close()

def change_color_violin(violin_parts):
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = violin_parts[partname]
        vp.set_edgecolor('grey')

    for pc in violin_parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('grey')

def box_plot_distance_violin_subplots(path1, path2, path3,path4, path5, path6):
    import seaborn as sns
    #sns.set_style('whitegrid')
    #ax = sns.violinplot(x='Survived', y='Age', data=df)
    #ax = sns.stripplot(x="Survived", y="Age", data=df)
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 14,
            }
    matplotlib.rc('font', size=14)
    matplotlib.rc('axes', titlesize=14)
    plt.figure(figsize=(14, 14))
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    data4 = pd.read_csv(path4)
    data5 = pd.read_csv(path5)
    data6 = pd.read_csv(path6)
    d_1 = data1['Distance_1']
    d_2 = data2['Distance_1']
    d_3 = data3['Distance_1']
    d_4 = data4['Distance_1']
    d_5 = data5['Distance_1']
    d_6 = data6['Distance_1']

    fig, axs = plt.subplots(2,2)
    violin_parts=axs[0, 0].violinplot([d_1, d_2, d_3],[1, 2, 3],points=80, vert=True, widths=0.7,
                         showmeans=True, showextrema=True, showmedians=False)
    change_color_violin(violin_parts)
    axs[0, 0].set_xticks([1, 2, 3])
    axs[0, 0].set_xticklabels(['Our Approach', 'Wachter et al.', 'Van Looveren et al.'], fontsize=14)
    axs[0, 0].set_ylim([0, 1])
    axs[1, 0].set_xlabel('MNIST', fontsize=14)
    axs[0, 0].set_ylabel('l1', fontsize=14)
    violin_parts=axs[0, 1].violinplot([d_4, d_5, d_6], [1, 2, 3], points=80, vert=True, widths=0.7,
                         showmeans=True, showextrema=True, showmedians=False)
    change_color_violin(violin_parts)
    axs[0, 1].set_xticks([1, 2, 3])
    axs[0, 1].set_xticklabels(['Our Approach', 'Wachter et al.', 'Van Looveren et al.'], fontsize=14)
    axs[0, 1].set_ylim([0, 1])
    axs[1, 1].set_xlabel('Fashion', fontsize=14)

    violin_parts=axs[1, 0].violinplot([data1['Distance_2'], data2['Distance_2'],
                   data3['Distance_2']],[1, 2, 3],points=80, vert=True, widths=0.7,
                   showmeans=True, showextrema=True, showmedians=False)
    change_color_violin(violin_parts)
    axs[1, 0].set_xticks([1, 2, 3])
    axs[1, 0].set_xticklabels(['Our Approach', 'Wachter et al.', 'Van Looveren et al.'], fontsize=14)
    axs[1, 0].set_ylim([0, 1])
    axs[1, 0].set_ylabel('l2', fontsize=14)
    #axs[1, 0].set_title('MNIST')

    violin_parts=axs[1, 1].violinplot([data4['Distance_2'], data5['Distance_2'],
                          data6['Distance_2']], [1, 2, 3], points=80, vert=True, widths=0.7,
                         showmeans=True, showextrema=True, showmedians=False)
    change_color_violin(violin_parts)
    axs[1, 1].set_xticks([1, 2, 3])
    axs[1, 1].set_xticklabels(['Our Approach', 'Wachter et al.', 'Van Looveren et al.'], fontsize=14)
    axs[1, 1].set_ylim([0, 1])
    #axs[1, 1].set_title('Fashion')

    plt.show()
    #plt.savefig(f'./Benchmark/dist_violin_full.png', transparent=True, bbox_inches='tight')
    plt.close()



#def success_rate(path, model):
#    import pickle5 as pickle

#    success=0
#    for pic in os.listdir(path):
#        if not pic.endswith('.csv') and not pic.endswith('.txt'):
#            for file in os.listdir(f'{path}/{pic}'):
#                if file.startswith('BestHof'):
#                    target=re.search(r'\d', file)
#                    cf=pickle.load(open(f'{path}/{pic}/{file}','rb'))
#                    actual= model.predict(helper.reshape_to_image(np.array(cf[0]), (28, 28, 1)).reshape(1,28,28,1))
#                    print(target[0])
#                    print(np.argmax(actual))
#                    if str(target[0]) == str(np.argmax(actual)):
#                        success=success+1
#    print('Success', success/90)

def allPics(dir1, dir2, dir3, key='None',save='Test.png'):
    '''
    Picture wise comparison of the apprioaches
    dir1: Path to results from our approach
    dir2: Path to results from Alibi for Wachter et al.
    dir3: Path to results from Alibi for Van Looveren et al.
    '''
    #getOverallBestPics(dir1,1)
    x = 1
    plt.figure(figsize=(14, 8))
    files = [f for f in os.listdir(dir1)]
    files = sorted(files)
    print(files)
    map = []
    for sample in files:
        print(str(dir1) + '/' + str(sample) + '/')
        if not sample.endswith('.txt') and not sample.endswith('.csv'):
            for f in os.listdir(str(dir1) + '/' + str(sample) + '/'):
                if 'beginningIm' in f:
                    filename = f
                    output = re.search(r'\d', filename)
                    print(output[0])
            infile = open(str(dir1) + '/' + str(sample) + '/' + filename, 'rb')
            images = pickle.load(infile)
            infile.close()
            plt.subplot(4, 10, int(output[0]) + 1)
            if (int(output[0]) + 1 == 1):
                plt.ylabel('Original', fontdict=font)
            map.append([output[0], sample])
            image = np.array(images).reshape(28, 28)
            plt.imshow(image, cmap='gray')
            if key == 'None':
                plt.title('Originally ' + str(output[0]), fontdict=font)
            else:
                plt.title(key[int(output[0])], fontdict=font)
            plt.xticks([])
            plt.yticks([])
            x = x + 1
    # Wachter et al.
    map = pd.DataFrame(map, columns=['class', 'pic'])
    files = [f for f in os.listdir(dir2)]
    files = sorted(files)
    print(files)
    base = x
    for sample in files:
        #print('Sample ',sample)
        infile = open(str(dir2) + '/' + str(sample) + '/explain_noTarget.pkl', 'rb')
        image = pickle.load(infile)
        infile.close()
        id = map[map['pic'] == sample]
        #print(id)
        #print(sample)
        #print('res ',id['class'].values)
        plt.subplot(4, 10, int(id['class'].values[0]) + base)
        image.cf['X'].reshape(28, 28)
        plt.imshow(image.cf['X'].reshape(28, 28), cmap='gray')
        if key == 'None':
            plt.title('Classified ' + str(image.cf['class']), fontdict=font)
        else:
            plt.title(key[image.cf['class']], fontdict=font)
        if (int(id['class']) + base == 11):
            plt.ylabel('Wachter et al.', fontdict=font)
        plt.xticks([])
        plt.yticks([])

        x = x + 1
    # VanLooven.
    files = [f for f in os.listdir(dir3)]
    files = sorted(files)
    print(files)
    base = x
    for sample in files:
        infile = open(str(dir3) + '/' + str(sample) + '/explain_noTarget.pkl', 'rb')
        image = pickle.load(infile)
        infile.close()
        id = map[map['pic'] == sample]
        plt.subplot(4, 10, int(id['class']) + base)
        image.cf['X'].reshape(28, 28)
        plt.imshow(image.cf['X'].reshape(28, 28), cmap='gray')
        if key == 'None':
            plt.title('Classified ' + str(image.cf['class']), fontdict=font)
        else:
            plt.title(key[image.cf['class']], fontdict=font)
        if (int(id['class']) + base == 21):
            plt.ylabel('Van Looveren et al.', fontdict=font)
        plt.xticks([])
        plt.yticks([])
        x = x + 1

    files = [f for f in os.listdir(dir1)]
    files = sorted(files)
    base = x
    for sample in files:
        if not sample.endswith('.txt') and  not sample.endswith('.csv') :
            infile = open(str(dir1) + '/' + str(sample) + '/Best_1_Images.pkl', 'rb')
            images = pickle.load(infile)
            infile.close()
            id = map[map['pic'] == sample]
            plt.subplot(4, 10, int(id['class']) + base)
            rand = random.randint(0, len(images) - 1)
            print(rand)
            images = images[rand]
            image = np.array(images)
            image=helper.reshape_to_image(image, (28,28,1)).reshape(28, 28,1)
            plt.imshow(image, cmap='gray')
            if (int(id['class']) + base == 31):
                plt.ylabel('Our Approach', fontdict=font)
            print(images.output)
            if key == 'None':
                plt.title('Classified ' + str(np.argmax(images.output)), fontdict=font)
            else:
                plt.title(key[np.argmax(images.output)], fontdict=font)
            plt.xticks([])
            plt.yticks([])
        # plt.title(images.input)
            x = x + 1
    plt.savefig(save, transparent=True, bbox_inches='tight')

def averaged(path1, path2, path3):
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    df=pd.DataFrame([])
    df['knn_mean']=[np.mean(data1['y-Nearest-Neighbours']),np.mean(data2['y-Nearest-Neighbours']),np.mean(data3['y-Nearest-Neighbours'])]
    df['knn_std'] = [np.std(data1['y-Nearest-Neighbours']),np.std(data2['y-Nearest-Neighbours']),np.std(data3['y-Nearest-Neighbours'])]
    df['redundancy'] = [np.mean(data1['Redundancy']),np.mean(data2['Redundancy']),np.mean(data3['Redundancy'])]
    df['redundancy_std'] = [np.std(data1['Redundancy']),np.std(data2['Redundancy']),np.std(data3['Redundancy'])]
    df['dis_1'] = [np.mean(data1['Distance_1']), np.mean(data2['Distance_1']), np.mean(data3['Distance_1'])]
    df['dis_1_std'] = [np.std(data1['Distance_1']), np.std(data2['Distance_1']), np.std(data3['Distance_1'])]
    df['dis_2'] = [np.mean(data1['Distance_2']), np.mean(data2['Distance_2']), np.mean(data3['Distance_2'])]
    df['dis_2_std'] = [np.std(data1['Distance_2']), np.std(data2['Distance_2']), np.std(data3['Distance_2'])]

    return df

def picture_example_added_and_deleted_pixels(path, save='image.png'):

    for folder in os.listdir(f'{path}'):
        if not folder.startswith('command') and not folder.endswith('.csv'):
            for file in os.listdir(f'{path}/{folder}'):
                if file.startswith('Best_1_'):
                    individual = pickle.load(open(f'{path}/{folder}/{file}','rb'))
                    target = np.argmax(individual[0].output)
                    pic= helper.reshape_to_image(np.array(individual[0]), (28,28,1)).reshape(28,28)
                    cf=pic
                if file.startswith('beginning'):
                    original= pickle.load(open(f'{path}/{folder}/{file}','rb'))
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
                plt.imshow(original.reshape(28, 28, 1),cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'{path}/{folder}/Original_{folder}.png', transparent=True, bbox_inches='tight')
                plt.close()

                plt.figure()
                plt.autoscale(tight='True')
                plt.imshow(pic.reshape(28, 28, 1),cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'{path}/{folder}/CF_{folder}.png', transparent=True, bbox_inches='tight')
                plt.close()

                plt.figure()
                plt.autoscale(tight='True')
                plt.imshow(orig_add.reshape(28, 28, 3))
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'{path}/{folder}/Added_{folder}.png',transparent=True, bbox_inches='tight')
                plt.close()

                plt.figure()
                plt.autoscale(tight='True')
                plt.imshow(orig_delete.reshape(28, 28, 3))
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'{path}/{folder}/Deleted_{folder}.png',transparent=True, bbox_inches='tight')
                plt.close()

def ssim_closest_image(path, save='image.png', data  = 'MNIST'):
    #distances=['ssim','issm','fsim','rmse','me']
    if data == 'MNIST':
        x_train, y_train, _ ,_= helper.load_mnist()
    elif data == 'Fashion':
        x_train, y_train, _, _ = helper.load_fashion_mnist()
    for folder in os.listdir(f'{path}'):
        if not folder.startswith('command') and not folder.endswith('csv'):
            for file in os.listdir(f'{path}/{folder}'):
                if file.startswith('Best_1_'):
                    print(f'{path}/{folder}')
                    individual = pickle.load(open(f'{path}/{folder}/{file}','rb'))
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
                    plt.savefig(f'{path}//{folder}/SSIM_{folder}.png', transparent=True, bbox_inches='tight')
                    plt.close()





if __name__ == '__main__':
    import pickle5 as pickle
    import tensorflow as tf
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    # MNIST
    #df=averaged('./Benchmark/Evo/mnist_me/benchmark_results.csv','./Alibi/Wachter/MNIST/Classwise/benchmark_results.csv','./Alibi/VanLooveren/MNIST/Classwise/benchmark_results.csv')
    #df.to_csv('./Benchmark/Evo/mnist_me/Averaged.csv')
    #box_plot_distance('./Benchmark/Evo/mnist_me/benchmark_results.csv','./Alibi/Wachter/MNIST/Classwise/benchmark_results.csv','./Alibi/VanLooveren/MNIST/Classwise/benchmark_results.csv',save='MNIST')
    #picture_example_added_and_deleted_pixels('./Benchmark/Evo/mnist_me/')
    #ssim_closest_image('./Benchmark/Evo/mnist_me/', data='MNIST')
    #box_plot_distance_violin('./Benchmark/Evo/mnist_me/benchmark_results.csv',
    #                  './Alibi/Wachter/MNIST/Classwise/benchmark_results.csv',
    #                  './Alibi/VanLooveren/MNIST/Classwise/benchmark_results.csv', save='MNIST')
    #allPics('./Benchmark/Evo/mnist_me', './Alibi/Wachter/MNIST/Direction_Other', './Alibi/VanLooveren/MNIST/Direction_Other',
    #        save='./Benchmark/MNIST.png')

    #Fashion
    #df=averaged('./Benchmark/Evo/fashion_me/benchmark_results.csv', './Alibi/Wachter/Fashion/Classwise/benchmark_results.csv',
    #         './Alibi/VanLooveren/Fashion/Classwise/benchmark_results.csv')
    #df.to_csv('./Benchmark/Evo/fashion_me/Averaged.csv')
    #picture_example_added_and_deleted_pixels('./Benchmark/Evo/fashion_me/')
    #ssim_closest_image('./Benchmark/Evo/fashion_me/', data='Fashion')
    #box_plot_distance('./Benchmark/Evo/fashion_me/benchmark_results.csv',
    #                  './Alibi/Wachter/Fashion/Classwise/benchmark_results.csv',
    #                  './Alibi/VanLooveren/Fashion/Classwise/benchmark_results.csv', save='Fashion')
    #box_plot_distance_violin('./Benchmark/Evo/fashion_me/benchmark_results.csv',
    #                                    './Alibi/Wachter/Fashion/Classwise/benchmark_results.csv',
    #                                    './Alibi/VanLooveren/Fashion/Classwise/benchmark_results.csv', save='Fashion')
    #key = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #allPics('./Benchmark/Evo/fashion_me', './Alibi/Wachter/Fashion/Direction_Other',
    #        './Alibi/VanLooveren/Fashion/Direction_Other',key=key,
    #        save='./Benchmark/Fashion.png')
    box_plot_distance_violin_subplots('./Benchmark/Evo/mnist_me/benchmark_results.csv',
                      './Alibi/Wachter/MNIST/Classwise/benchmark_results.csv',
                      './Alibi/VanLooveren/MNIST/Classwise/benchmark_results.csv','./Benchmark/Evo/fashion_me/benchmark_results.csv',
                                        './Alibi/Wachter/Fashion/Classwise/benchmark_results.csv',
                                        './Alibi/VanLooveren/Fashion/Classwise/benchmark_results.csv')