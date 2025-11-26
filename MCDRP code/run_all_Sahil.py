# import Model_unified_same_zone_matching_Sahil_changed as unified_same_zone
#import Model_unified_same_zone_matching_Sahil_original as unified_same_zone

import Model_unified_same_zone_matching_Sahil as unified_same_zone

# Original testing
# for inst_string in ['_imbalanced']:
# new testing larger instances (10/06/22, N = 50)
# make sure to refresh data and "same_zone_matching" code
supply_type_list, inst_list, cnt_m = [0], range(1), 0  # range(10)
beta, enter_exit = 1, False
for N in [30]:  # [0.5, 2]:
    if beta == 0.5:
        beta_string = '_half_beta'
    elif beta == 1:
        beta_string = '_one_beta'
    elif beta == 2:
        beta_string = '_double_beta'
    data_name = 'data/chicago-synthetic-inst-N%d_v4.npz' % N
    # for enter_exit in [False, True]:
    model_type_list, disturb_list, ee_str = ['UB'], [False], ''  # if enter_exit else ['simple', 'LB']

    # todo: run testing for heatmap, not ee, dist inst 1-4
    for model_type in model_type_list:
        if enter_exit and model_type == 'LB':
            continue
        for disturb in disturb_list:
            dist_str = '_disturb' if disturb else ''
            inst_string = ee_str + '_' + model_type + beta_string + dist_str
            unified_same_zone.solve_sequentially(inst_string, inst_list, beta, model_type, True, data_name,
                                     enter_exit, disturb, cnt_m)
