import tempfile
import os

import numpy as np
from numpy import array

from cmme.drex.base import UnprocessedPrior, DistributionType, GaussianPrior
from cmme.drex.binding import from_mat, to_mat, DREXInstructionsFile, DREXResultsFile, DREXResultsFilePsi, \
    transform_to_estimatesuffstat_representation, transform_to_rundrexmodel_representation
from cmme.drex.model import DREXInstructionBuilder
from cmme.drex.util import transform_to_unified_drex_input_sequence_representation
from cmme.drex import DREXModel


def prior_input_sequence_from_mat_to_trialtimefeature_sequence(prior_input_sequence_from_mat):
    data = prior_input_sequence_from_mat[0][0]
    trial_count = data.shape[0]
    res = list()
    for trial_idx in range(trial_count):
        res.append(data[trial_idx].T.tolist())
    return transform_to_unified_drex_input_sequence_representation(res)


def test_to_mat_and_from_mat():
    data = {
        "a": "b",
        "c": [1, 2, 3],
        "d": float('inf')
    }

    tmp_file_path = tempfile.NamedTemporaryFile().name

    to_mat(data, tmp_file_path)
    data_after_save_and_read = from_mat(tmp_file_path, simplify_cells=False)

    assert data["a"] == data_after_save_and_read["a"][0]
    assert data["c"] == data_after_save_and_read["c"][0].tolist()
    assert data["d"] == data_after_save_and_read["d"][0]


def test_drex_instructions_file():
    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3, 4, 5])
    prior_input_sequence = transform_to_unified_drex_input_sequence_representation([[[11, 12, 13, 14, 15]], [[21, 22, 23, 24, 25]]])
    prior = UnprocessedPrior(DistributionType.GMM, prior_input_sequence)
    hazard = 0.1
    memory = float('inf')
    maxhyp = 2
    obsnz = 0.1
    predscale = 0.001
    beta = 0.002
    max_ncomp = 11
    change_decision_threshold = 0.1
    instructions_file = DREXInstructionsFile(input_sequence, prior, hazard, memory, maxhyp, obsnz,
                                             max_ncomp, beta,
                                             predscale, change_decision_threshold)

    tmp_file_path = tempfile.NamedTemporaryFile().name

    instructions_file.save_self(tmp_file_path)

    data_after_read = from_mat(tmp_file_path, simplify_cells=False)
    assert np.array_equal(transform_to_estimatesuffstat_representation(prior_input_sequence), transform_to_estimatesuffstat_representation(prior_input_sequence_from_mat_to_trialtimefeature_sequence(data_after_read["estimate_suffstat"]["xs"][0])))
    assert prior.distribution_type().value == data_after_read["estimate_suffstat"]["params"][0][0][0]["distribution"][0][0]
    assert prior.D_value() == data_after_read["estimate_suffstat"]["params"][0][0][0]["D"][0][0][0]
    assert np.array_equal(transform_to_rundrexmodel_representation(instructions_file.input_sequence), data_after_read["run_DREX_model"]["x"][0][0])
    assert prior.distribution_type().value == data_after_read["run_DREX_model"]["params"][0][0][0]["distribution"][0][0]
    assert prior.D_value() == data_after_read["run_DREX_model"]["params"][0][0][0]["D"][0][0][0]
    assert instructions_file.hazard == data_after_read["run_DREX_model"]["params"][0][0][0]["hazard"][0][0][0]
    assert instructions_file.memory == data_after_read["run_DREX_model"]["params"][0][0][0]["memory"][0][0][0]
    assert instructions_file.maxhyp == data_after_read["run_DREX_model"]["params"][0][0][0]["maxhyp"][0][0][0]
    assert instructions_file.obsnz == data_after_read["run_DREX_model"]["params"][0][0][0]["obsnz"][0][0][0]
    assert max_ncomp == data_after_read["run_DREX_model"]["params"][0][0][0]["max_ncomp"][0][0][0]
    assert beta == data_after_read["run_DREX_model"]["params"][0][0][0]["beta"][0][0][0]
    assert instructions_file.change_decision_threshold == data_after_read["post_DREX_changedecision"]["threshold"][0][0][0]


def test_drex_results_file():
    instructions_file_path = "drex-instructionsfile-gaussian-D2.mat"
    input_sequence = array([[1., 1., 1.],
                            [1., 1., 2.],
                            [1., 1., 3.],
                            [1., 1., 3.],
                            [1., 1., 2.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 2.],
                            [1., 1., 3.],
                            [1., 1., 3.],
                            [1., 1., 2.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 2.],
                            [1., 1., 3.],
                            [1., 1., 3.],
                            [1., 2., 2.],
                            [1., 2., 1.],
                            [1., 2., 1.],
                            [1., 2., 2.],
                            [1., 1., 3.],
                            [1., 1., 3.],
                            [1., 1., 2.],
                            [1., 1., 1.]])
    means = array([[2.5, 2.5],
                   [3.5, 3.5],
                   [4.5, 4.5]])
    covariance = array([[[1.25, 0.41666667],
                         [0.41666667, 1.25]],
                        [[1.25, 0.41666667],
                         [0.41666667, 1.25]],
                        [[1.25, 0.41666667],
                         [0.41666667, 1.25]]])
    n = array([2, 2, 2])
    prior = GaussianPrior(means, covariance, n)
    surprisal = array([[13.12548487, 14.59244541, 15.78916213],
                       [11.62806218, 11.89348203, 11.20732747],
                       [11.22211153, 11.37462962, 11.04539885],
                       [10.96862618, 11.08944953, 11.05768208],
                       [10.78294117, 10.89033817, 13.92801269],
                       [10.63646819, 10.73686459, 13.55070915],
                       [10.51489474, 10.6111308, 11.47971593],
                       [10.41040257, 10.50404047, 11.43222382],
                       [10.31955706, 10.41143807, 11.66713045],
                       [10.23906182, 10.32966324, 10.9844449],
                       [10.16655485, 10.25620026, 12.30249722],
                       [10.10069016, 10.18958236, 12.83508243],
                       [10.04026424, 10.12852552, 11.29397161],
                       [9.9843672, 10.07211162, 11.42938704],
                       [9.93258458, 10.01991213, 11.81596062],
                       [9.88414354, 9.9710681, 10.98322152],
                       [9.83847074, 13.86956707, 11.97942407],
                       [9.79864115, 10.08246164, 12.6332376],
                       [9.75714687, 10.03744203, 11.2279844],
                       [9.71862759, 9.99611845, 11.40974472],
                       [9.68233091, 14.16792606, 11.89624613],
                       [9.64994752, 10.14561185, 10.98842642],
                       [9.61536896, 10.10999454, 11.8171308],
                       [9.58353385, 10.07742609, 12.53696054]])
    joint_surprisal = array([43.50709241, 34.71745457, 33.62830658, 33.10059997, 35.59853852,
                             34.91454479, 32.58568605, 32.32783597, 32.38243523, 31.53442929,
                             32.70721028, 33.10785988, 31.44099014, 31.46580723, 31.7523164,
                             30.8184713, 35.70261124, 32.49620538, 31.00083883, 31.10474571,
                             35.75060371, 30.76216833, 31.52146667, 32.1780187])
    context_beliefs = array([[1.00000000e+00, 9.90000000e-01, 9.89328126e-01, 9.89005473e-01,
                              9.88987232e-01, 9.87939286e-01, 9.86868880e-01, 9.87891679e-01,
                              9.88418852e-01, 9.88186340e-01, 9.87975232e-01, 9.87348957e-01,
                              9.86148057e-01, 9.85073680e-01, 9.84155565e-01, 9.82398909e-01,
                              9.80153632e-01, 9.84494948e-01, 9.84359243e-01, 9.83384357e-01,
                              9.82372384e-01, 9.82641128e-01, 9.82889822e-01, 9.81302741e-01,
                              9.78634069e-01],
                             [0.00000000e+00, 1.00000000e-02, 6.71873704e-04, 5.30153735e-04,
                              4.83590420e-04, 5.98416197e-04, 8.04304982e-04, 9.58977881e-04,
                              1.03872831e-03, 1.27306728e-03, 1.58316799e-03, 2.15269912e-03,
                              3.13513676e-03, 4.19201989e-03, 5.16360330e-03, 6.81041081e-03,
                              8.99902895e-03, 3.57073351e-03, 4.70624582e-03, 5.87419009e-03,
                              6.89852625e-03, 5.20735205e-03, 6.38615662e-03, 7.96380577e-03,
                              1.03936331e-02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 4.64373334e-04,
                              1.84817870e-04, 1.94612997e-04, 2.86723054e-04, 2.72934133e-04,
                              1.83004477e-04, 1.70939220e-04, 1.80662280e-04, 2.31243261e-04,
                              3.54999224e-04, 4.48498644e-04, 4.40128967e-04, 4.99066237e-04,
                              6.02261394e-04, 2.90338719e-04, 3.95814188e-04, 4.82011963e-04,
                              4.89158105e-04, 3.81821767e-04, 4.39710922e-04, 5.26193523e-04,
                              7.01147796e-04],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02,
                              3.44359475e-04, 3.43908707e-04, 6.65462631e-04, 3.80316029e-04,
                              1.21228850e-04, 7.71644152e-05, 6.99327291e-05, 8.71971048e-05,
                              1.40655503e-04, 1.68852582e-04, 1.29552290e-04, 1.19908684e-04,
                              1.33362515e-04, 8.11395150e-05, 1.14132957e-04, 1.36794356e-04,
                              1.20401987e-04, 9.35478862e-05, 1.01685463e-04, 1.20956376e-04,
                              1.64381738e-04],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              1.00000000e-02, 9.23776197e-04, 9.31579401e-04, 3.01633347e-04,
                              7.24012040e-05, 3.69029367e-05, 2.60586983e-05, 2.87159228e-05,
                              4.25835044e-05, 4.37280814e-05, 3.00234152e-05, 2.51725154e-05,
                              2.46525276e-05, 1.80812351e-05, 2.46422950e-05, 2.70485972e-05,
                              2.22622548e-05, 1.85701698e-05, 1.84967707e-05, 2.12373858e-05,
                              2.83045860e-05],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 1.00000000e-02, 4.43049883e-04, 8.55227385e-05,
                              1.75347256e-05, 8.02371964e-06, 4.67726122e-06, 3.51217536e-06,
                              3.95612846e-06, 3.52146915e-06, 2.25203687e-06, 1.85274632e-06,
                              1.68605580e-06, 1.29168321e-06, 1.51869020e-06, 1.54188542e-06,
                              1.20855395e-06, 1.14840296e-06, 1.09660409e-06, 1.09591026e-06,
                              1.31518611e-06],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 1.08936501e-04,
                              2.00351506e-05, 9.09893029e-06, 4.51962355e-06, 1.58934003e-06,
                              1.00394482e-06, 7.47087493e-07, 4.93472520e-07, 4.15290554e-07,
                              3.62148417e-07, 2.77090833e-07, 2.47994742e-07, 2.29080743e-07,
                              1.84354723e-07, 2.04439065e-07, 1.91018039e-07, 1.64914887e-07,
                              1.64436626e-07],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02,
                              1.28215570e-04, 4.26589640e-05, 1.31408003e-05, 2.19164870e-06,
                              8.42019232e-07, 4.54069193e-07, 2.70594466e-07, 2.14280983e-07,
                              1.62231222e-07, 1.40365332e-07, 1.07910551e-07, 8.58745032e-08,
                              6.65573244e-08, 8.22914094e-08, 7.04582644e-08, 5.66819620e-08,
                              5.13296484e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              1.00000000e-02, 1.95804481e-04, 2.63169141e-05, 3.73496311e-06,
                              1.57293487e-06, 6.77201911e-07, 2.49819038e-07, 1.50772059e-07,
                              9.70111829e-08, 1.01277305e-07, 8.25515771e-08, 6.24043850e-08,
                              3.87259540e-08, 4.70005283e-08, 3.66745006e-08, 2.80497701e-08,
                              2.61665939e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 1.00000000e-02, 1.16291423e-04, 1.56531697e-05,
                              8.65797735e-06, 2.23794598e-06, 3.92477213e-07, 1.61413874e-07,
                              8.90593326e-08, 1.16315335e-07, 1.00313770e-07, 7.25097202e-08,
                              3.53849088e-08, 3.99440078e-08, 2.86466664e-08, 2.16193669e-08,
                              2.07115779e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 1.24505989e-04,
                              3.58902928e-05, 5.25591911e-06, 6.94094497e-07, 2.28585707e-07,
                              9.82688851e-08, 1.46070273e-07, 1.16796668e-07, 7.27605995e-08,
                              3.19922770e-08, 3.71703347e-08, 2.33871835e-08, 1.66169756e-08,
                              1.53395786e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02,
                              1.26644483e-04, 1.10567488e-05, 1.24723507e-06, 3.68757121e-07,
                              1.30867396e-07, 1.71411954e-07, 1.05221412e-07, 5.73431529e-08,
                              2.36771513e-08, 3.05391556e-08, 1.77651731e-08, 1.02809567e-08,
                              8.13423147e-09],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              1.00000000e-02, 4.92703466e-05, 4.98549517e-06, 1.46292263e-06,
                              4.42393207e-07, 3.53295976e-07, 1.23306736e-07, 5.68231699e-08,
                              2.44749403e-08, 3.62900645e-08, 2.00837647e-08, 9.03221861e-09,
                              5.39663994e-09],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 1.00000000e-02, 7.05414304e-05, 1.51645361e-05,
                              2.84390974e-06, 1.41951176e-06, 3.06973063e-07, 1.04007568e-07,
                              4.09121547e-08, 6.34242913e-08, 3.01445669e-08, 1.18902974e-08,
                              6.04172709e-09],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 1.26513817e-04,
                              1.03520285e-05, 5.94220683e-06, 1.44542872e-06, 3.98238739e-07,
                              9.84079095e-08, 1.26462810e-07, 5.01927994e-08, 1.84352963e-08,
                              9.80325011e-09],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02,
                              7.07983039e-05, 5.57390828e-05, 1.84259399e-05, 3.11024807e-06,
                              3.70967997e-07, 3.37107815e-07, 1.11187813e-07, 3.90895723e-08,
                              2.16051354e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              1.00000000e-02, 1.47906039e-03, 2.45348126e-04, 2.30001546e-05,
                              2.04767796e-06, 1.34753005e-06, 3.24205172e-07, 9.76212304e-08,
                              4.87393071e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 1.00000000e-02, 1.31992829e-04, 8.97319117e-06,
                              8.15448070e-07, 8.89407186e-07, 2.09810774e-07, 5.15047029e-08,
                              2.33544423e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 5.78332077e-05,
                              4.81353106e-06, 6.72664606e-06, 1.33465303e-06, 1.54042301e-07,
                              3.94204936e-08],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02,
                              8.74670189e-05, 1.17373498e-04, 1.41572815e-05, 7.80966141e-07,
                              1.22708090e-07],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              1.00000000e-02, 1.52909011e-03, 7.83371346e-05, 3.74827252e-06,
                              6.55993383e-07],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 1.00000000e-02, 6.80885321e-05, 4.02887863e-06,
                              1.16973706e-06],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 5.47325708e-05,
                              8.28175771e-06],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02,
                              6.64777314e-05],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              1.00000000e-02]])
    belief_dynamics = array([0.00000000e+00, 1.81243858e-05, 3.52673839e-03, 3.86735645e-03,
                             4.12840797e-03, 3.18315516e-03, 3.95404765e-03, 4.87115785e-03,
                             4.72279393e-03, 4.43527560e-03, 4.67277399e-03, 4.65437157e-03,
                             4.67382049e-03, 4.91240628e-03, 4.79206544e-03, 4.67181823e-03,
                             4.89123736e-03, 3.48214178e-03, 4.98457815e-03, 4.95131071e-03,
                             4.74213521e-03, 2.60324151e-03, 5.39773500e-03, 4.90944098e-03,
                             4.88967249e-03])
    change_decision_changepoint = 2
    change_decision_probability = array([0., 0.01, 0.01067187, 0.01099453, 0.01101277,
                                         0.01206071, 0.01313112, 0.01210832, 0.01158115, 0.01181366,
                                         0.01202477, 0.01265104, 0.01385194, 0.01492632, 0.01584443,
                                         0.01760109, 0.01984637, 0.01550505, 0.01564076, 0.01661564,
                                         0.01762762, 0.01735887, 0.01711018, 0.01869726, 0.02136593])
    change_decision_threshold = 0.01
    psi = DREXResultsFilePsi({}, {})  # TODO

    rf = DREXResultsFile(instructions_file_path,
                         input_sequence, prior, surprisal, joint_surprisal,
                         context_beliefs, belief_dynamics, change_decision_changepoint,
                         change_decision_probability, change_decision_threshold, psi)

    assert rf is not None


def test_results_file_reads_file_without_error():
    rf1 = DREXResultsFile.load(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-gaussian-D1.mat"))
    rf2 = DREXResultsFile.load(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-gaussian-D2.mat"))
    rf3 = DREXResultsFile.load(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-gmm-D1.mat"))
    rf4 = DREXResultsFile.load(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-poisson-D3.mat"))


def test_load_drex_instructions_file():
    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3, 4, 5])
    prior_input_sequence = transform_to_unified_drex_input_sequence_representation(
        [[[11, 12, 13, 14, 15]], [[21, 22, 23, 24, 25]]])
    prior = UnprocessedPrior(DistributionType.GMM, prior_input_sequence)
    hazard = 0.1
    memory = float('inf')
    maxhyp = 2
    obsnz = 0.1
    predscale = 0.001
    beta = 0.002
    max_ncomp = 11
    change_decision_threshold = 0.1
    instructions_file = DREXInstructionsFile(input_sequence, prior, hazard, memory, maxhyp, obsnz,
                                             max_ncomp, beta,
                                             predscale, change_decision_threshold)

    tmp_file_path = tempfile.NamedTemporaryFile().name

    instructions_file.save_self(tmp_file_path)

    test_instructions_file = DREXInstructionsFile.load(tmp_file_path)
    assert (test_instructions_file.input_sequence == input_sequence).all()
    assert (test_instructions_file.prior.prior_input_sequence == prior.prior_input_sequence).all()
    assert test_instructions_file.prior.distribution_type() == prior.distribution_type()
    assert test_instructions_file.prior.D_value() == prior.D_value()
    assert test_instructions_file.prior.feature_count() == prior.feature_count()
    assert test_instructions_file.prior.trials_count() == prior.trials_count()
    assert test_instructions_file.hazard == hazard
    assert test_instructions_file.memory == memory
    assert test_instructions_file.maxhyp == maxhyp
    assert test_instructions_file.obsnz == [obsnz]
    assert test_instructions_file.max_ncomp == max_ncomp
    assert test_instructions_file.beta == beta
    assert test_instructions_file.predscale == predscale
    assert test_instructions_file.change_decision_threshold == change_decision_threshold

def test_load_results_file_GMM():
    prior_distribution_type = DistributionType.GMM
    prior_input_sequence = [1, 1, 1, 1, 4, 4, 4, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    prior_D = 1
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    drex_instance = DREXInstructionBuilder()

    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.input_sequence(input_sequence)
    drex_instance.hazard(hazard)
    drex_instance.memory(memory)
    drex_instance.maxhyp(maxhyp)
    drex_instance.obsnz(obsnz)
    drex_instance.change_decision_threshold(change_decision_threshold)
    drex_instance.prior(prior)
    with tempfile.TemporaryDirectory() as tmpdirname:
        instructions_file_path = tmpdirname + "-instructionsfile"
        results_file_path = tmpdirname + "-resultsfile"

        drex_model = DREXModel()
        drex_instance.to_instructions_file() \
            .save_self(instructions_file_path, results_file_path)
        results_file = drex_model.run(instructions_file_path)

        assert results_file.prior.distribution_type() == DistributionType.GMM
        assert results_file.prior.D_value() == prior_D
        assert results_file.prior.feature_count() == 1
        assert (results_file.input_sequence == input_sequence).all()

def test_load_results_file_Gaussian():
    prior_distribution_type = DistributionType.GAUSSIAN
    prior_input_sequence = [1, 1, 1, 1, 4, 4, 4, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    prior_D = 2
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    drex_instance = DREXInstructionBuilder()

    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.input_sequence(input_sequence)
    drex_instance.hazard(hazard)
    drex_instance.memory(memory)
    drex_instance.maxhyp(maxhyp)
    drex_instance.obsnz(obsnz)
    drex_instance.change_decision_threshold(change_decision_threshold)
    drex_instance.prior(prior)
    with tempfile.TemporaryDirectory() as tmpdirname:
        instructions_file_path = tmpdirname + "-instructionsfile"
        results_file_path = tmpdirname + "-resultsfile"

        drex_model = DREXModel()
        drex_instance.to_instructions_file() \
            .save_self(instructions_file_path, results_file_path)
        results_file = drex_model.run(instructions_file_path)

        assert results_file.prior.distribution_type() == DistributionType.GAUSSIAN
        assert results_file.prior.D_value() == prior_D
        assert results_file.prior.feature_count() == 1
        assert (results_file.input_sequence == input_sequence).all()

def test_load_results_file_Lognormal():
    prior_distribution_type = DistributionType.LOGNORMAL
    prior_input_sequence = [1, 1, 1, 1, 4, 4, 4, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    prior_D = 2
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    drex_instance = DREXInstructionBuilder()

    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.input_sequence(input_sequence)
    drex_instance.hazard(hazard)
    drex_instance.memory(memory)
    drex_instance.maxhyp(maxhyp)
    drex_instance.obsnz(obsnz)
    drex_instance.change_decision_threshold(change_decision_threshold)
    drex_instance.prior(prior)
    with tempfile.TemporaryDirectory() as tmpdirname:
        instructions_file_path = tmpdirname + "-instructionsfile"
        results_file_path = tmpdirname + "-resultsfile"

        drex_model = DREXModel()
        drex_instance.to_instructions_file() \
            .save_self(instructions_file_path, results_file_path)
        results_file = drex_model.run(instructions_file_path)

        assert results_file.prior.distribution_type() == DistributionType.LOGNORMAL
        assert results_file.prior.D_value() == prior_D
        assert results_file.prior.feature_count() == 1
        assert (results_file.input_sequence == input_sequence).all()

def test_load_results_file_Lognormal():
    prior_distribution_type = DistributionType.POISSON
    prior_input_sequence = [1, 1, 1, 1, 4, 4, 4, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    prior_D = 2
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    drex_instance = DREXInstructionBuilder()

    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.input_sequence(input_sequence)
    drex_instance.hazard(hazard)
    drex_instance.memory(memory)
    drex_instance.maxhyp(maxhyp)
    drex_instance.obsnz(obsnz)
    drex_instance.change_decision_threshold(change_decision_threshold)
    drex_instance.prior(prior)
    with tempfile.TemporaryDirectory() as tmpdirname:
        instructions_file_path = tmpdirname + "-instructionsfile"
        results_file_path = tmpdirname + "-resultsfile"

        drex_model = DREXModel()
        drex_instance.to_instructions_file() \
            .save_self(instructions_file_path, results_file_path)
        results_file = drex_model.run(instructions_file_path)

        assert results_file.prior.distribution_type() == DistributionType.POISSON
        assert results_file.prior.feature_count() == 1
        assert (results_file.input_sequence == input_sequence).all()