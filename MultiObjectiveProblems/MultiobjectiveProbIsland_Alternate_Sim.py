import autograd.numpy as anp
import numpy as np
import numpy
from pymop.problem import Problem
import cv2
import math
import Additional_Help_Functions as helper
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.framework import ops
import sklearn
#import image_similarity_measures.quality_metrics as metrics
# always derive from the main problem for the evaluation

from skimage.metrics import structural_similarity as ssim
import phasepack.phasecong as pc


#Read for information on similarity measures: https://www.tandfonline.com/doi/full/10.1080/22797254.2019.1628617

class MyProblem(Problem):

    def __init__(self,model, equation_inputs, input_group,procid, measure):

        # define lower and upper bounds -  1d array with length equal to number of variable
        #print(equation_inputs.shape)
        xl = anp.zeros(equation_inputs.shape[0]*equation_inputs.shape[1]*equation_inputs.shape[2])
        xu = 255* anp.ones(equation_inputs.shape[0]*equation_inputs.shape[1]*equation_inputs.shape[2])

        super().__init__(n_var=equation_inputs.shape[0]*equation_inputs.shape[1]*equation_inputs.shape[2], n_obj=3, n_constr=0, xl=xl, xu=xu, evaluation_of="auto")
        # store custom variables needed for evaluation
        self.measure= getattr(self,measure)
        self.procid=procid
        self.equation_inputs = equation_inputs
        self.input_group=input_group
        self.model =model
        dim = int(np.sqrt(self.n_var))
        self.output=self.model.predict(self.equation_inputs.reshape(1,equation_inputs.shape[0],equation_inputs.shape[1],equation_inputs.shape[2])/255)

    def ssim(self,org,pop):
        return (1 - ssim(org.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],3), pop.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],3), data_range=pop.max() - pop.min(),multichannel=True)) / 2

    def me(self,org,pop):
        return numpy.mean(abs(numpy.subtract(org.reshape(1, -1), pop.reshape(1, -1))),axis = 1 )/255

    def rmse(self, org, pop):
        return np.sqrt(np.mean(np.square( org.reshape(-1)/255- pop.reshape(-1)/255)))

    def _ehs(self,x, y):
        """
        Entropy-Histogram Similarity measure
        """
        H = (np.histogram2d(x.flatten(), y.flatten()))[0]
        #print(H)
        return -np.sum(np.nan_to_num(H * np.log2(H)))

    def _edge_c(self,x, y):
        """
        Edge correlation coefficient based on Canny detector
        """
        # Use 100 and 200 as thresholds, no indication in the paper what was used
        g = cv2.Canny(np.uint8(x), 100, 200)
        h = cv2.Canny(np.uint8(y), 100, 200)
        # g = cv2.Canny((x * 0.0625).astype(np.uint8), 100, 200)
        # h = cv2.Canny((y * 0.0625).astype(np.uint8), 100, 200)

        g0 = np.mean(g)
        h0 = np.mean(h)

        numerator = np.sum((g - g0) * (h - h0))
        denominator = np.sqrt(np.sum(np.square(g - g0)) * np.sum(np.square(h - h0)))

        return numerator / denominator

    def calculate_issm(self,org_img: np.ndarray, pred_img: np.ndarray) -> float:
        """
        Information theoretic-based Statistic Similarity Measure

        Note that the term e which is added to both the numerator as well as the denominator is not properly
        introduced in the paper. We assume the authers refer to the Euler number.
        """
        #_assert_image_shapes_equal(org_img, pred_img, "ISSM")

        # Variable names closely follow original paper for better readability
        x = org_img
        y = pred_img
        A = 0.3
        B = 0.5
        C = 0.7

        ehs_val = self._ehs(x, y)
        canny_val = self._edge_c(x, y)

        numerator = canny_val * ehs_val * (A + B) + math.e
        #print('x', type(x))
        #print('y',type(y))
        denominator = A * canny_val * ehs_val + B * ehs_val + C * ssim(x, y,multichannel=True) + math.e

        return np.nan_to_num(numerator / denominator)
    def issm(self,org,pop):
        #TODO Currently Sometimes Throws devision by zero
        issm = 1-self.calculate_issm(org.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],3),
                     pop.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],
                                 3))
        return issm

    def _similarity_measure(self,x, y, constant):
        """
        Calculate feature similarity measurement between two images
        """
        numerator = 2 * x * y + constant
        denominator = x ** 2 + y ** 2 + constant

        return numerator / denominator

    def _gradient_magnitude(self,img: np.ndarray, img_depth):
        """
        Calculate gradient magnitude based on Scharr operator
        """
        scharrx = cv2.Scharr(img, img_depth, 1, 0)
        scharry = cv2.Scharr(img, img_depth, 0, 1)

        return np.sqrt(scharrx ** 2 + scharry ** 2)

    def calculate_fsim(self, org_img: np.ndarray, pred_img: np.ndarray, T1=0.85, T2=160) -> float:
        """
        Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)

        There are different ways to implement PC, the authors of the original FSIM paper use the method
        defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
        of the approach.

        There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
        operation which is implemented in OpenCV.

        Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
        are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
        band and then take the average.

        Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
        would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.

        Args:
            org_img -- numpy array containing the original image
            pred_img -- predicted image
            T1 -- constant based on the dynamic range of PC values
            T2 -- constant based on the dynamic range of GM values
        """
        #_assert_image_shapes_equal(org_img, pred_img, "FSIM")

        alpha = beta = 1  # parameters used to adjust the relative importance of PC and GM features
        fsim_list = []
        for i in range(org_img.shape[2]):
            # Calculate the PC for original and predicted images
            pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
            pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

            # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
            # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
            # calculate the sum of all these 6 arrays.
            pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
            pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
            for orientation in range(6):
                pc1_2dim_sum += pc1_2dim[4][orientation]
                pc2_2dim_sum += pc2_2dim[4][orientation]

            # Calculate GM for original and predicted images based on Scharr operator
            gm1 = self._gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
            gm2 = self._gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

            # Calculate similarity measure for PC1 and PC2
            S_pc = self._similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
            # Calculate similarity measure for GM1 and GM2
            S_g = self._similarity_measure(gm1, gm2, T2)

            S_l = (S_pc ** alpha) * (S_g ** beta)

            numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
            denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
            fsim_list.append(numerator / denominator)

        return np.mean(fsim_list)

    def fsim(self,org,pop):
        fsim=1-self.calculate_fsim(np.float32(org.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],3)), np.float32(pop.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],3)))
        return fsim

    def _evaluate(self, x,out, *args, **kwargs):

        pop = numpy.array(x)


        pop = pop.reshape(1, self.equation_inputs.shape[0], self.equation_inputs.shape[1], self.equation_inputs.shape[2])
        #print(pop.shape)
        d = self.model.predict(pop / 255)
        #print('model predict finished')
        c = d.argmax(axis=1)
        if self.equation_inputs.shape[2]== 1:
            pop=pop.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1])
            equation_inputs= self.equation_inputs.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1])
            distance = self.measure(np.array([equation_inputs,equation_inputs,equation_inputs]),np.array([pop,pop,pop]))
        else:
            distance = self.measure(self.equation_inputs,pop.reshape(self.equation_inputs.shape[0], self.equation_inputs.shape[1],self.equation_inputs.shape[2]))

        substract = abs(numpy.subtract(self.equation_inputs.reshape(1, -1), pop.reshape(1, -1)))
        sparsity = numpy.count_nonzero(substract)
        output_distance =d[0][self.procid]

        if(c==self.input_group):
            output_distance=0
            sparsity=self.equation_inputs.shape[0]*self.equation_inputs.shape[1]*self.equation_inputs.shape[2]
            distance=1
        out["F"] = np.column_stack([distance, sparsity/(self.equation_inputs.shape[0]*self.equation_inputs.shape[1]*self.equation_inputs.shape[2]),(1-output_distance)])

    def number_objectives(self):
        return 3

    def header(self):
        return['distance','sparsity','output']

    def logging(self, tools):
        distance = tools.Statistics(lambda ind: ind.fitness.values[0])
        sparsity = tools.Statistics(lambda ind: ind.fitness.values[1])
        output = tools.Statistics(lambda ind: ind.fitness.values[2])

        mstats = tools.MultiStatistics(distance=distance,sparsity=sparsity ,output=output)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = "gen", "deme", "evals", "distance","sparsity", "output"
        logbook.chapters["distance"].header = "std", "min", "avg", "max"
        logbook.chapters["sparsity"].header = "std", "min", "avg", "max"
        logbook.chapters["output"].header = "std", "min", "avg", "max"

        return logbook, mstats