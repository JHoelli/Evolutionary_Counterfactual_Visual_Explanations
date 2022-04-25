import random
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import re
from Utils import getFitnessFromPareto, getOverallBestPics,getOverallBestPicsOutput, getOverallBestPicsDistance
from deap import base
import pandas as pd
from deap import creator
from Alibi_Utils import calculateData,last,counterfactualonly

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

def allPics(dir1, dir2, dir3, key='None',save='Test.png'):
    '''
    Picture wise comparison of the apprioaches
    dir1: Path to results from our approach
    dir2: Path to results from Alibi for Wachter et al.
    dir3: Path to results from Alibi for Van Looveren et al.
    '''
    getOverallBestPics(dir1,1)
    x = 1
    plt.figure(figsize=(14, 8))
    files = [f for f in os.listdir(dir1)]
    files = sorted(files)
    print(files)
    map = []
    for sample in files:
        print(str(dir1) + '/' + str(sample) + '/')
        if not sample.endswith('.txt'):
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
        infile = open(str(dir2) + '/' + str(sample) + '/explain_noTarget.pkl', 'rb')
        image = pickle.load(infile)
        infile.close()
        id = map[map['pic'] == sample]
        print(id)
        print(sample)
        print(id['class'].values[0])
        plt.subplot(4, 10, int(id['class']) + base)
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
        if not sample.endswith('.txt'):
            infile = open(str(dir1) + '/' + str(sample) + '/Best_1_Images.pkl', 'rb')
            images = pickle.load(infile)
            infile.close()
            id = map[map['pic'] == sample]
            plt.subplot(4, 10, int(id['class']) + base)
            rand = random.randint(0, len(images) - 1)
            print(rand)
            images = images[rand]
            image = np.array(images).reshape(28, 28)
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

def allPics_Output(dir1, dir2, dir3, key='None',save='Test.png'):
    '''
    Picture wise comparison of the apprioaches
    dir1: Path to results from our approach
    dir2: Path to results from Alibi for Wachter et al.
    dir3: Path to results from Alibi for Van Looveren et al.
    '''
    getOverallBestPicsDistance(dir1,1)
    x = 1
    plt.figure(figsize=(14, 8))
    files = [f for f in os.listdir(dir1)]
    files = sorted(files)
    print(files)
    map = []
    for sample in files:
        print(str(dir1) + '/' + str(sample) + '/')
        if not sample.endswith('.txt'):
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
        infile = open(str(dir2) + '/' + str(sample) + '/explain_noTarget.pkl', 'rb')
        image = pickle.load(infile)
        infile.close()
        id = map[map['pic'] == sample]
        print(id)
        print(sample)
        print(id['class'].values[0])
        plt.subplot(4, 10, int(id['class']) + base)
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
        if not sample.endswith('.txt'):
            infile = open(str(dir1) + '/' + str(sample) + '/Best_Distance_Overall_Images.pkl', 'rb')
            images = pickle.load(infile)
            infile.close()
            id = map[map['pic'] == sample]
            plt.subplot(4, 10, int(id['class']) + base)
            rand = random.randint(0, len(images) - 1)
            print(np.array(images).shape)
            print(images[rand])
            print(rand)
            images = images[rand]
            print(np.array(images).shape)
            image = np.array(images).reshape(28, 28)
            plt.imshow(image, cmap='gray')
            if (int(id['class']) + base == 31):
                plt.ylabel('Our Approach', fontdict=font)
            #print(images.output)
            if key == 'None':
                plt.title('Classified ' + str(np.argmax(images.output)), fontdict=font)
            else:
                plt.title(key[np.argmax(images.output)], fontdict=font)
            plt.xticks([])
            plt.yticks([])
        # plt.title(images.input)
            x = x + 1
    plt.savefig(save, transparent=True, bbox_inches='tight')

def allPics_Random(dir1, dir2, dir3, key='None',save='Test.png'):
    '''
    Picture wise comparison of the apprioaches
    dir1: Path to results from our approach
    dir2: Path to results from Alibi for Wachter et al.
    dir3: Path to results from Alibi for Van Looveren et al.
    '''
    #getOverallBestPicsOutput(dir1,1)
    x = 1
    plt.figure(figsize=(14, 8))
    files = [f for f in os.listdir(dir1)]
    files = sorted(files)
    print(files)
    map = []
    for sample in files:
        print(str(dir1) + '/' + str(sample) + '/')
        if not sample.endswith('.txt'):
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
        infile = open(str(dir2) + '/' + str(sample) + '/explain_noTarget.pkl', 'rb')
        image = pickle.load(infile)
        infile.close()
        id = map[map['pic'] == sample]
        print(id)
        print(sample)
        print(id['class'].values[0])
        plt.subplot(4, 10, int(id['class']) + base)
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
        if not sample.endswith('.txt'):
            for name in os.listdir(str(dir1) + '/' + str(sample)):
                if name.startswith('Output'):
                    rando=random.randint(0, 10)
                    try:
                        infile = open(str(dir1) + '/' + str(sample) + '/Output_Best_Island_'+str(rando)+'_1_Images.pkl', 'rb')
                    except:
                        if rando !=0:
                            infile = open(str(dir1) + '/' + str(sample) + '/Output_Best_Island_' + str(rando-1) + '_1_Images.pkl',
                                  'rb')

                        else:
                            infile = open(str(dir1) + '/' + str(sample) + '/Output_Best_Island_' + str(
                                rando + 1) + '_1_Images.pkl',
                                          'rb')
                    images = pickle.load(infile)
                    infile.close()
            id = map[map['pic'] == sample]
            plt.subplot(4, 10, int(id['class']) + base)
            rand = random.randint(0, len(images) - 1)
            print(np.array(images).shape)
            print(images[rand])
            print(rand)
            images = images[rand]
            print(np.array(images).shape)
            image = np.array(images).reshape(28, 28)
            plt.imshow(image, cmap='gray')
            if (int(id['class']) + base == 31):
                plt.ylabel('Our Approach', fontdict=font)
            #print(images.output)
            if key == 'None':
                plt.title('Classified ' + str(np.argmax(images.output)), fontdict=font)
            else:
                plt.title(key[np.argmax(images.output)], fontdict=font)
            plt.xticks([])
            plt.yticks([])
        # plt.title(images.input)
            x = x + 1
    plt.savefig(save, transparent=True, bbox_inches='tight')

def statisticalEvaluation(dir1, dir2, dir3,save_path='statistical_evaluation.csv'):
    x = 1
    path = dir1

    # Evaluation Alibi
    files = [f for f in os.listdir(dir2)]
    files = sorted(files)
    print(files)
    a_dist = []
    a_output = []
    a_sparsity = []
    for sample in files:
        # need exact path to file
        for sub in sample:
            if not os.path.exists(dir2 + '/' + str(sample) + '/' + str(sub) + '/' + 'explain_Target.pkl'):
                break;
            for ori in os.listdir(dir1 + '/' + sample):
                if 'beginning' in ori:
                    of = ori
            dis, spa, out = last(dir2 + '/' + str(sample) + '/' + str(sub) + '/' + 'explain_Target.pkl',
                                 dir1 + '/' + str(sample) + '/' + of)
            for d in dis:
                a_dist.append(d)
            for s in spa:
                a_sparsity.append(s)
            for o in out:
                a_output.append(o)

    # Evaluation Alibi --> Looveren
    files = [f for f in os.listdir(dir3)]
    files = sorted(files)
    print(files)
    b_dist = []
    b_output = []
    b_sparsity = []
    for sample in files:
        # need exact path to file
        for ori in os.listdir(dir1 + '/' + sample):
            if 'beginning' in ori:
                of = ori
        dis, spa, out = counterfactualonly(dir3 + '/' + str(sample), dir1 + '/' + str(sample) + '/' + of)
        for d in dis:
            b_dist.append(d)
        for s in spa:
            b_sparsity.append(s)
        for o in out:
            b_output.append(o)

    # Evaluation own Approach
    agg_dist = []
    agg_output = []
    agg_sparsity = []
    files = [f for f in os.listdir(path)]
    files = sorted(files)
    print(files)
    for sample in files:
        if not sample.endswith('.txt'):
            print(path + '/' + sample)
            infile = open(path + '/' + str(sample) + '/' + 'Best_1_Images.pkl', 'rb')
            new_dict = pickle.load(infile)
            infile.close()
            distance, sparsity, output = getFitnessFromPareto(new_dict)
            for a in distance:
                agg_dist.append(a)
            for b in sparsity:
                agg_sparsity.append(b)
            for c in output:
                agg_output.append(c)
    data = []
    data.append(
        ['Wachter et al.', np.mean(a_dist), np.std(a_dist), np.mean(a_sparsity), np.std(a_sparsity), np.mean(a_output),
         np.std(a_output)])
    data.append(
        ['Van Looveren', np.mean(b_dist), np.std(b_dist), np.mean(b_sparsity), np.std(b_sparsity),
         np.mean(b_output),
         np.std(b_output)])
    data.append(
        ['Our Approach', np.mean(agg_dist), np.std(agg_dist), np.mean(agg_sparsity), np.std(agg_sparsity),
         np.mean(agg_output),
         np.std(agg_output)])
    data = pd.DataFrame(data,
                        columns=['Method', 'Dist_Mean', 'Dist_Standarddevision', 'Spa_Mean', 'Spa_Standarddevision',
                                 'Out_Mean', 'Out_Standarddevision'])

    data.to_csv(save_path)

key=['T-shirt', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag', 'Ankle boot']
creator.create("FitnessMin", base.Fitness,weights=(-1.0,-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
'''mnist evauation'''
allPics('./MNIST','./Alibi/Wachter/MNIST/Direction_Other','./Alibi/VanLooveren/MNIST/Direction_Other',save='MNIST1.png')
statisticalEvaluation('./MNIST','./Alibi/Wachter/MNIST/Classwise','./Alibi/VanLooveren/MNIST/Classwise',save_path='Statistical_MNIST1.csv')
'''fashion evaluation'''
#allPics('./Fashion','./Alibi/Basic_Alibi_Fashion_Other/Fashion','./Alibi/Fashion_Best_Instances',key=key, save='Fashion.png')
#statisticalEvaluation('./Fashion','./Alibi/Alibi_Class_Wise_Fashion','./Alibi/Fashion_Counterfactual_Instances_Class_wise', save_path='Statistical_Fashion.csv')

allPics('./Fashion','./Alibi/Wachter/Fashion/Direction_Other','./Alibi/VanLooveren/Fashion/Direction_Other',key=key,save='Fashion1.png')
statisticalEvaluation('./Fashion','./Alibi/Wachter/Fashion/Classwise','./Alibi/VanLooveren/Fashion/Classwise',save_path='Statistical_Fashion1.csv')
#'./Alibi/Basic_Alibi_Fashion_Other/Fashion','./Alibi/Fashion_Best_Instances'